from __future__ import annotations

from typing import TYPE_CHECKING, List, NamedTuple, NoReturn, Set, Tuple, TypeAlias

import os

import torch
from freetoken.attention.linear import build_fla_metadata
from freetoken.core import Batch, Req
from freetoken.env import ENV
from freetoken.gpu_select import gpu_identity
from freetoken.message import (
    AbortBackendMsg,
    BaseBackendMsg,
    BatchBackendMsg,
    CacheRebuildBackendMsg,
    CacheRebuildResultMsg,
    DetokenizeMsg,
    ErrorReplyMsg,
    ExitMsg,
    PromptAdmittedMsg,
    UserMsg,
)
from freetoken.utils import (
    init_logger,
    load_eos_token_ids,
    load_tokenizer,
    load_toolcall_anchor_id,
)

from .cache import CacheManager
from .config import SchedulerConfig
from .decode import DecodeManager
from .io import SchedulerIOMixin
from .prefill import ChunkedReq, PrefillManager
from .status import SchedulerStatusReporter
from .table import TableManager

if TYPE_CHECKING:
    from freetoken.engine import BatchSamplingArgs, ForwardOutput


logger = init_logger(__name__)

Indice2D: TypeAlias = Tuple[torch.Tensor, torch.Tensor]


def _gib(n_bytes: int) -> str:
    return f"{n_bytes / (1 << 30):.2f} GiB"


# For overlap scheduling, we also need to cache some other data to avoid IMA
class ForwardInput(NamedTuple):
    batch: Batch
    sample_args: BatchSamplingArgs
    input_tuple: Indice2D  # (token_mapping, positions)
    write_tuple: Indice2D  # (req_mapping, seq_lens or -1)


ForwardData: TypeAlias = "Tuple[ForwardInput, ForwardOutput]"


class Scheduler(SchedulerIOMixin):
    def __init__(self, config: SchedulerConfig):
        from freetoken.engine import Engine

        self.engine = Engine(config)

        # use another stream to overlap metadata processing with computation
        self.device = self.engine.device
        self.stream = torch.cuda.Stream(device=self.device)
        self.engine_stream_ctx = torch.cuda.stream(self.engine.stream)
        torch.cuda.set_stream(self.stream)
        # sent on the readiness ack for /v1/stats gpus; a list so TP can add one entry per rank
        self.gpus = [gpu_identity(self.device.index)] if self.device.type == "cuda" else []

        # initialize other managers
        self.table_manager = TableManager(config.max_running_req, self.engine.page_table)
        # ONE cache manager for every model (ShadowRadix layering): the shared page table is the
        # virtual full-token coordinate; model-specific tiers ride the plug-ins -- DSV4's
        # window/cmp/idx shadows via swa_pool, Gemma's swa via swa_pool, GDN state via
        # linear_state_pool. No model supplies its own manager.
        self.cache_manager = CacheManager(
            self.engine.num_pages, config.page_size, self.engine.page_table, config.cache_type,
            linear_state_pool=self.engine.linear_state_pool,
            swa_pool=self.engine.kv_cache,
            sliding_window_size=next(
                (g.sliding_window for g in config.model_config.kv_cache_group_specs() if g.is_swa),
                None,
            ) or getattr(self.engine.kv_cache, "sliding_window_size", None),
        )
        self.decode_manager = DecodeManager(config.page_size)
        self.prefill_manager = PrefillManager(
            self.cache_manager, self.table_manager, self.decode_manager
        )

        # some alias for easy access
        self.finished_reqs: Set[Req] = set()
        # Abort acknowledgements are a terminal accounting barrier. Queue them while processing
        # inbound control messages, then flush only AFTER _process_last_data publishes any
        # sampled replies from the prior overlapped forward.
        self._pending_abort_acks: Set[int] = set()
        # With multiple tokenizer workers, an AbortBackendMsg and its earlier UserMsg can arrive
        # through different PUSH producers and be observed out of order. Preserve a bounded
        # tombstone so an abort-before-admission request can never be resurrected after its
        # terminal accounting acknowledgement has already been published.
        self._abort_tombstones: dict[int, None] = {}
        self._forward_iter = 0  # global forward counter; drives the SWA proactive-eviction cadence
        # The launched-but-not-yet-drained batch (overlap): set at the top of each overlap_loop
        # iteration so the abort handler can tell whether a request's forward is still in flight
        # (mark it, defer the free to _process_last_data) or not (free immediately). Stays None
        # in normal_loop, where a batch launches and drains within one iteration.
        self._last_data: ForwardData | None = None
        # A received-but-not-yet-executed runtime cache rebuild (CacheRebuildBackendMsg),
        # run at the next idle safe point in overlap_loop. None when no rebuild is pending.
        self._pending_rebuild: CacheRebuildBackendMsg | None = None
        self.tokenizer = load_tokenizer(config.model_path)
        self.eos_token_ids = load_eos_token_ids(config.model_path, self.tokenizer)
        self.toolcall_anchor_id = None
        if config.special_token_ckpt and (
            self.cache_manager.is_hybrid or self.cache_manager.is_swa
        ):
            from freetoken.server.function_call_parser import toolcall_opener_for

            self.toolcall_anchor_id = load_toolcall_anchor_id(
                self.tokenizer,
                toolcall_opener_for(getattr(config, "tool_call_parser", "")),
            )
        self.token_pool = self.table_manager.token_pool
        # Floor the prefill chunk by the cache manager's cap (DSV4: ~half the window pool) so a
        # sliding-window cache chunks long prompts and frees out-of-window pages between chunks
        # instead of OOMing _alloc_window on a prompt longer than the window pool.
        _chunk_cap = self.cache_manager.prefill_chunk_budget
        self.prefill_budget = (
            min(config.max_extend_tokens, _chunk_cap) if _chunk_cap else config.max_extend_tokens
        )
        self.config = config
        self.status_reporter = SchedulerStatusReporter(
            log=logger.info_rank0,
            decode_log_interval=config.decode_log_interval,
        )

        # Initialize the I/O mixin
        super().__init__(config, self.engine.tp_cpu_group)

    def run_when_idle(self) -> None:
        """Called when the scheduler is idle to perform background tasks."""
        logger.info_rank0("Scheduler is idle, waiting for new reqs...")
        self.cache_manager.check_integrity()

    @torch.inference_mode()
    def rebuild_cache(
        self,
        *,
        moe_cache_size: int | None = None,
        num_pages: int | None = None,
        num_mamba_slots: int | None = None,
        num_swa_pages: int | None = None,
    ) -> None:
        """Idle-only runtime cache rebuild: resize the MoE slot cache, KV pages, GDN (mamba) state
        pool, and/or the window pool (num_swa_pages), re-capture CUDA graphs, and re-thread the
        page managers (clearing the prefix cache on a KV/mamba/window resize). The caller MUST
        guarantee the scheduler is idle — no pending prefill, no running decode, no in-flight
        finished requests. All TP ranks must call this with identical arguments.
        """
        assert not self.prefill_manager.runnable, "rebuild requires no pending prefill"
        assert not self.decode_manager.runnable, "rebuild requires no running decode"
        torch.cuda.synchronize(self.device)
        if self.config.tp_info.size > 1:
            self.sync_all_ranks()
        self.engine.rebuild_runtime_cache(
            moe_cache_size=moe_cache_size, num_pages=num_pages, num_mamba_slots=num_mamba_slots,
            num_swa_pages=num_swa_pages,
        )
        if num_pages is not None or num_mamba_slots is not None or num_swa_pages is not None:
            # Any of these resizes invalidates the prefix cache: a KV resize leaves stale page
            # indices, a mamba resize leaves stale GDN-snapshot slot ids, and a window-pool resize
            # (num_swa_pages) reallocates the SWA/window token pool, leaving stale slot ids in the
            # radix tree. Rebuild the prefix cache + reclaim the resized free-lists.
            self.cache_manager.rebuild(self.engine.num_pages, self.engine.page_table)
            if num_pages is not None:
                # token_pool is sized to the page table; only a KV-page resize reallocates it.
                # A mamba-only rebuild leaves the page table untouched, so skip this (else it
                # needlessly reallocates + zeros the whole GPU token_pool every mamba resize).
                self.table_manager.rebuild(self.engine.page_table)
                self.token_pool = self.table_manager.token_pool
            self.cache_manager.check_integrity()
        # The prefill chunk cap tracks the CURRENT window-pool size (DSV4); a rebuild that
        # shrank the pool must shrink the cap too, or the next long prompt is chunked against
        # the stale budget and crashes _alloc_window.
        _chunk_cap = self.cache_manager.prefill_chunk_budget
        self.prefill_budget = (
            min(self.config.max_extend_tokens, _chunk_cap)
            if _chunk_cap else self.config.max_extend_tokens
        )
        if self.config.tp_info.size > 1:
            self.sync_all_ranks()

    def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
        """
        The main loop of overlapping scheduling and execution.

        It will overlap the execution of current batch and processing of last batch's results,
        which can effectively hide CPU latency and improve GPU utilization.
        """
        # Expose the un-drained batch to _process_one_msg (abort in-flight check). Assigning
        # before the message loop is what makes the check airtight: the batch launched later
        # this iteration can only be probed by messages of the NEXT iteration, which sees it here.
        self._last_data = last_data
        blocking = not (
            last_data is not None  # don't block if we have a batch to be processed
            or self.prefill_manager.runnable
            or self.decode_manager.runnable
            or self._pending_rebuild is not None  # a queued rebuild to drain toward + execute
        )
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        # Execute a queued cache rebuild once the scheduler is fully idle (the safe point):
        # no last batch to process, no pending prefill, no running decode. finished_reqs is
        # NOT a gate — those requests are already freed (no live GPU/page resources).
        if self._pending_rebuild is not None and last_data is None and not (
            self.prefill_manager.runnable or self.decode_manager.runnable
        ):
            self._execute_pending_rebuild()

        # Order this iteration's host->device token_pool copies (issued on ``self.stream``
        # during scheduling) after the previous batch's sampled-token writes (issued on the
        # engine stream in ``_forward``). Without this, a request that reuses a just-freed
        # table_idx can have its freshly copied prompt clobbered by the prior occupant's
        # still-pending output write -- corrupting tokens (e.g. dropping an image
        # placeholder, which the multimodal merge then rejects).
        self.stream.wait_stream(self.engine.stream)
        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            with self.engine_stream_ctx:  # run the batch in the engine's stream
                self.engine.stream.wait_stream(self.stream)
                # COW-restore GDN snapshots for prefix hits ON THE ENGINE STREAM, after the
                # cross-stream wait and before the forward reads the live slot (program order
                # vs the prior batch's snapshot writes). Doing this on self.stream would race.
                self._restore_linear_states(forward_input.batch)
                ongoing_data = (forward_input, self._forward(forward_input))

        # The drain issues GPU-visible writes to state the batch just launched still reads: the
        # page-table re-point and, for the paged-SWA pools, the full->swa (DSV4: full->window)
        # sentinel scatter. DSV4 stages the page table at replay time and translates
        # full_to_window INSIDE the captured graph, so an unordered drain can redirect an
        # in-flight forward. copy_done only covers batch N; order against N+1 explicitly.
        self.stream.wait_stream(self.engine.stream)
        self._process_last_data(last_data)
        self._flush_abort_acks()
        return ongoing_data

    def normal_loop(self) -> None:
        blocking = not (
            self.prefill_manager.runnable
            or self.decode_manager.runnable
            or self._pending_rebuild is not None  # a queued rebuild to execute at idle
        )
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        # Non-overlap mode has no last_data to drain; execute a queued rebuild as soon as
        # the scheduler is idle (no pending prefill / running decode). Without this, a
        # rebuild in DISABLE_OVERLAP_SCHEDULING mode stays pending until the HTTP timeout.
        if self._pending_rebuild is not None and not (
            self.prefill_manager.runnable or self.decode_manager.runnable
        ):
            self._execute_pending_rebuild()

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            # already inside engine_stream_ctx (run_forever); restore on the engine stream
            self._restore_linear_states(forward_input.batch)
            ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(ongoing_data)
        self._flush_abort_acks()

    @torch.inference_mode()
    def run_forever(self) -> NoReturn:
        # DSV4 (owned-KV) decode reads its per-token window/cmp/idx slot maps off the attention
        # backend's per-batch SNAPSHOT (staged in prepare_for_replay right before the replay, on
        # the same stream, like the generic out_loc copy_from), not the live slot maps -- so the
        # next batch's allocate_paged cannot corrupt the in-flight graph replay. DSV4 overlaps.
        if ENV.DISABLE_OVERLAP_SCHEDULING:
            with self.engine_stream_ctx:
                self.engine.stream.wait_stream(self.stream)
                while True:
                    self.normal_loop()
        else:
            assert torch.cuda.current_stream() == self.stream
            data = None
            while True:
                data = self.overlap_loop(data)

    def shutdown(self) -> None:
        torch.cuda.synchronize(self.device)
        self.sync_all_ranks()
        self.engine.shutdown()

    def _process_last_data(self, last_data: ForwardData | None) -> None:
        if last_data is None:
            return

        batch = last_data[0].batch
        _, next_tokens_cpu, copy_done, spec_extra_cpu = last_data[1]
        copy_done.synchronize()
        # MTP spec decode: extra emissions per req beyond next_tokens (the
        # FIRST emitted token). [B] int32, -1-padded on reject; None on
        # plain batches. Emission order per req: first token, then the
        # bonus (accept) — with the FULL per-token check sequence looped
        # in order (append_host -> length -> EOS -> stop-str -> toolcall
        # anchor), stopping at the first finish.
        reply: List[DetokenizeMsg] = []
        new_finished_reqs: Set[Req] = set()
        with self.cache_manager.lazy_free_region():
            for i, req in enumerate(batch.reqs):
                if isinstance(req, ChunkedReq):
                    # Don't cache intermediate chunks; the full prompt is cached once when the
                    # final chunk is processed. Caching here snapshots a handle the next chunk
                    # already copied (overlap), so cache_req double-frees the prior chunk.
                    if req.aborted:
                        # Aborted mid-chunked-prefill while this chunk was in flight: the abort
                        # popped the pending continuation (no next chunk launches), and this
                        # drain point frees the chunk's pages/slots exactly once.
                        self._free_req_resources(req)
                    continue
                if req.aborted:
                    # Aborted while this final-chunk prefill / decode step was in flight: free
                    # here (the forward is drained) and finish the request. No DetokenizeMsg --
                    # the abort ack flushed after this method stays the uid's terminal reply.
                    self.decode_manager.remove_req(req)
                    self._free_req_resources(req)
                    new_finished_reqs.add(req)
                    continue
                if req in self.finished_reqs:
                    # Overlap scheduling launched one more decode step for a request that
                    # already terminated (filter_reqs keeps it while output budget remains,
                    # and the next batch is scheduled before this drain runs). Its resources
                    # are freed below/already; shipping this token would append past the
                    # client's terminal reply.
                    continue
                emitted = [next_tokens_cpu[i]]
                if spec_extra_cpu is not None:
                    for tok in spec_extra_cpu[i]:
                        if int(tok.item()) >= 0:
                            emitted.append(tok)
                # report_batch counted the default 1/req; count the spec
                # bonus emission(s) on top for an exact throughput window.
                self.status_reporter.count_generated_tokens(len(emitted) - 1)
                if os.environ.get("FT_DEBUG_STORE"):
                    import sys as _sys
                    print(f"[drain] uid={req.uid} emitted={[int(t.item()) for t in emitted]}",
                          file=_sys.stderr)
                finished = False
                for tok in emitted:
                    if spec_extra_cpu is None:
                        # Plain decode: append_host is THE host mirror of
                        # the GPU token write. Spec batches must NOT
                        # append: the emitted tokens are already placed in
                        # the ids view at their KV positions (the draft at
                        # q+1 by _prepare_spec_batch; the bonus becomes
                        # the next iteration's staged input). Appending
                        # would shift them +1 vs their KV positions.
                        req.append_host(tok.unsqueeze(0))
                    token = int(tok.item())
                    # EOS / stop-string -> "stop", output budget exhausted ->
                    # "length"; EOS and stop strings win over length. The
                    # length check re-runs PER TOKEN (multi-token emission
                    # consumes 2 budget units; the exact per-token cutoff).
                    hit_length = not req.can_decode
                    hit_eos = (
                        not req.sampling_params.ignore_eos
                        and token in self.eos_token_ids
                    )
                    matched_stop = (
                        self._match_stop_str(req)
                        if not hit_eos and req.sampling_params.stop_strs
                        else None
                    )
                    finished = hit_length or hit_eos or matched_stop is not None
                    finish_reason = (
                        ("stop" if (hit_eos or matched_stop is not None) else "length")
                        if finished
                        else None
                    )
                    if (
                        token == self.toolcall_anchor_id
                        and req.toolcall_anchor_len is None
                        and not finished
                    ):
                        req.toolcall_anchor_len = req.input_ids.numel()
                    reply.append(
                        DetokenizeMsg(
                            uid=req.uid,
                            next_token=token,
                            finished=finished,
                            finish_reason=finish_reason,
                            matched_stop=matched_stop,
                            stop_strs=req.sampling_params.stop_strs or None,
                        )
                    )
                    if finished:
                        break

                # NOTE: overlap scheduling may make the request freed twice, skip second free
                if finished and req not in self.finished_reqs:
                    self.decode_manager.remove_req(req)
                    self._free_req_resources(req)
                    new_finished_reqs.add(req)
                elif batch.is_prefill and req.table_idx != -1:
                    # for prefill, non-chunk req, cache the prefix.
                    # Polymorphic: the DSV4 naive manager keeps the request's slots (no-op);
                    # the generic manager inserts the prefix into its radix/naive cache.
                    # table_idx == -1 is defense-in-depth: aborts mark in-flight requests
                    # instead of freeing them (handled above), so a freed request should
                    # never reach this commit -- but if a future path frees one early, skip
                    # rather than re-read the freed page-table row (and on hybrid, deref the
                    # None'd GDN ping-pong slots).
                    self.cache_manager.cache_req(req, finished=False)

        self.finished_reqs = new_finished_reqs
        # Stamp each reply with the post-batch KV page occupancy so the frontend (shell
        # status bar) can show live KV usage without a separate query.
        used, total = self._kv_usage_pages()
        mamba_slots = self._mamba_slot_usage()
        swa_tokens = self._swa_token_usage()
        if reply:
            mem = self._gpu_mem_bytes()
            mamba_used, mamba_total = mamba_slots or (0, 0)
            swa_used, swa_total = swa_tokens or (0, 0)
            for m in reply:
                m.kv_used_pages = used
                m.kv_total_pages = total
                m.mamba_used_slots = mamba_used
                m.mamba_total_slots = mamba_total
                m.swa_used_tokens = swa_used
                m.swa_total_tokens = swa_total
                m.gpu_mem_bytes = mem
        self.status_reporter.report_batch(
            batch,
            running_reqs=len(self.decode_manager.running_reqs),
            queue_reqs=len(self.prefill_manager.pending_list),
            kv_used_pages=used,
            kv_total_pages=total,
            page_size=self.config.page_size,
            mamba_slots=mamba_slots,
            swa_tokens=swa_tokens,
        )
        self.send_result(reply)

    def _match_stop_str(self, req: Req) -> str | None:
        """First stop string present in this request's generated tail, else None. Decodes
        only a short suffix (bounded by the longest stop string's char length, so a stop of
        N chars spans at most N tokens) to keep the per-step cost small."""
        stop_strs = req.sampling_params.stop_strs
        prompt_len = req.max_device_len - req.output_len
        if len(req.input_ids) <= prompt_len:
            return None
        max_chars = max(len(s) for s in stop_strs)
        tail_start = max(prompt_len, len(req.input_ids) - (max_chars + 1))
        tail = self.tokenizer.decode(req.input_ids[tail_start:].tolist())
        for s in stop_strs:
            if s in tail:
                return s
        return None

    def _kv_usage_pages(self) -> Tuple[int, int]:
        """(used_pages, total_pages) of the KV page pool.

        ``used`` follows SGLang's logging semantics: allocated pages that are not
        evictable (active requests + protected prefix cache). Evictable prefix-cache
        pages are available to future requests, so they are excluded from usage.
        Always the manager's own primary pool (for DSV4 the FULL cmp/idx tier); the
        window (swa) tier is reported separately by ``_swa_token_usage``.
        """
        return self.cache_manager.page_usage()

    def _mamba_slot_usage(self) -> Tuple[int, int] | None:
        """(used_slots, total_slots) of the GDN-state (mamba) pool for hybrid models, else None.

        Mirrors SGLang's mamba-pool semantics: ``total`` excludes the reserved padding
        sink (slot 0); ``used`` excludes free slots and evictable tree snapshots.
        """
        if not self.cache_manager.is_hybrid:
            return None
        total = self.cache_manager.linear_state_pool.num_slots - 1
        return total - self.cache_manager.mamba_available_size, total

    def _swa_token_usage(self) -> Tuple[int, int] | None:
        """(used_tokens, total_tokens) of the window (swa) pool for SWA models, else None.

        Mirrors the mamba accounting: ``total`` excludes the pool's reserved sentinel
        unit; ``used`` excludes free slots and evictable (unlocked) tree tokens.
        """
        cm = self.cache_manager
        if not cm.swa_paged:
            return None
        total = cm.swa_pool.swa_num_tokens - 1
        return total - cm.swa_available_size, total

    def _gpu_mem_bytes(self) -> int:
        """Bytes this engine process holds on the GPU (torch's reserved caching-allocator
        pool: weights + KV + MoE cache + graphs). 0 on CPU. Cheap, no device sync."""
        if self.device.type != "cuda":
            return 0
        return torch.cuda.memory_reserved(self.device)

    def _process_one_msg(self, msg: BaseBackendMsg) -> None:
        if isinstance(msg, BatchBackendMsg):
            for msg in msg.data:
                self._process_one_msg(msg)
        elif isinstance(msg, ExitMsg):
            raise KeyboardInterrupt
        elif isinstance(msg, UserMsg):
            logger.debug_rank0("Received user msg: %s", msg)
            tombstones = getattr(self, "_abort_tombstones", None)
            if tombstones is not None and msg.uid in tombstones:
                tombstones.pop(msg.uid, None)
                logger.debug_rank0(
                    "Dropping request %d because its abort arrived before admission", msg.uid
                )
                return
            input_len, max_seq_len = len(msg.input_ids), self.engine.max_seq_len
            max_output_len = max_seq_len - input_len
            if max_output_len <= 0:
                logger.warning_rank0(
                    f"Input sequence length {input_len} exceeds {max_seq_len}, "
                    f"request {msg.uid} is dropped."
                )
                # Tell the client instead of dropping silently — otherwise its wait_for_ack
                # never sees a `finished` reply and hangs until the request times out.
                self.send_result(
                    [
                        ErrorReplyMsg(
                            uid=msg.uid,
                            # "prompt is too long: N tokens > M" is the phrasing Claude Code and
                            # OpenClaw match on; the Anthropic wire has no error code to read.
                            error=(
                                f"prompt is too long: {input_len} tokens > {max_seq_len} maximum "
                                f"(prompt + generation); shorten the prompt or increase the KV "
                                f"cache budget"
                            ),
                            # OpenAI's standard class for this, for clients that read a code.
                            code="context_length_exceeded",
                        )
                    ]
                )
                return
            if msg.sampling_params.max_tokens > max_output_len:
                msg.sampling_params.max_tokens = max_output_len
                logger.warning_rank0(
                    f"Adjust max_tokens to {max_output_len} for request {msg.uid}."
                )
            self.prefill_manager.add_one_req(msg)
        elif isinstance(msg, AbortBackendMsg):
            logger.debug_rank0("Aborting request %d", msg.uid)
            tombstones = getattr(self, "_abort_tombstones", None)
            if tombstones is None:
                tombstones = self._abort_tombstones = {}
            tombstones[msg.uid] = None
            # Unknown aborts normally consume their tombstone when the cross-worker UserMsg
            # catches up. Bound hostile/no-followup abort traffic without affecting realistic
            # in-flight concurrency.
            while len(tombstones) > 65_536:
                tombstones.pop(next(iter(tombstones)))
            req_to_free = self.prefill_manager.abort_req(msg.uid)
            req_to_free = req_to_free or self.decode_manager.abort_req(msg.uid)
            if req_to_free is not None:
                # SGLang-style abort: never free resources under an in-flight forward. If the
                # request is in the launched-but-not-drained batch (overlap), only mark it;
                # _process_last_data frees it this same iteration, after copy_done.synchronize()
                # -- so its KV pages / GDN slots are never recycled mid-write, and the
                # finished=False prefix-commit can't run on a freed request. A request with no
                # forward in flight (e.g. a decode req starved behind a long chunked prefill)
                # is freed immediately -- deferring would leak until its next batch, which
                # strict prefill-priority puts arbitrarily far away.
                inflight = (
                    self._last_data is not None
                    and req_to_free in self._last_data[0].batch.reqs
                )
                if inflight:
                    req_to_free.aborted = True
                else:
                    self._free_req_resources(req_to_free)
            # Always acknowledge the abort, even when the request already left the manager,
            # but NOT yet: overlap_loop still has to publish the prior forward's sampled reply.
            # _flush_abort_acks runs after _process_last_data, making this a true terminal
            # accounting barrier for FrontendManager/prepare-stop.
            self._pending_abort_acks.add(msg.uid)
        elif isinstance(msg, CacheRebuildBackendMsg):
            # v1 scope: only if_idle, single-rank, non-owned-KV. drain mode and TP rebuild
            # need the drain-gate / all-rank failure-agreement machinery (deferred), so we
            # reject them cleanly rather than ship hang-prone half-wired paths.
            if not self.cache_manager.supports_runtime_rebuild:
                self._reply_rebuild(
                    msg.request_id, "unsupported", "this model's cache does not support runtime rebuild"
                )
            elif msg.mode != "if_idle":
                self._reply_rebuild(
                    msg.request_id, "unsupported", f"mode {msg.mode!r} unsupported (use if_idle)"
                )
            elif self.config.tp_info.size > 1:
                self._reply_rebuild(
                    msg.request_id, "unsupported", "runtime rebuild unsupported under TP > 1"
                )
            elif self.prefill_manager.runnable or self.decode_manager.runnable:
                # if_idle: refuse rather than wait. (finished_reqs hold no resources — they
                # are already freed — so they do not block a rebuild.)
                self._reply_rebuild(msg.request_id, "busy")
            else:
                self._pending_rebuild = msg
        else:
            logger.error(f"Unknown message type: {type(msg)}")
            raise NotImplementedError

    def _restore_linear_states(self, batch) -> None:
        """COW-restore a hybrid prefix hit's GDN snapshot into its freshly-allocated live slot
        (first chunk only). MUST run on the ENGINE stream so it is program-ordered after the
        prior batch's snapshot writes and before this forward reads the live slot."""
        pool = self.engine.linear_state_pool
        if pool is None or not batch.is_prefill:
            return
        for req in batch.reqs:
            if req.mamba_restore_src is not None:
                pool.copy_from(req.mamba_restore_src, req.linear_slot_idx)
                req.mamba_restore_src = None  # consumed: restore exactly once

    def _free_req_resources(self, req: Req) -> None:
        # Idempotent: an EOS-finished request can stay in running_reqs (output budget left), so an
        # abort in the same overlap iteration races _process_last_data and would free it twice --
        # double-freeing its table_idx and (hybrid) GDN slots onto the free-list, handing the same
        # slots to two later requests. table_idx == -1 marks an already-freed request.
        if req.table_idx == -1:
            return
        # MTP spec tail (defect-1 fix): pages allocate_paged mapped for an
        # in-flight spec iteration's verify rows are mapped-but-uncommitted;
        # the finish paths below free/donate [0, cached_len) exclusively, so
        # an abort/finish landing between _prepare_spec_batch and the resolve
        # would leak them ("free_pages + cache_pages != num_pages"). The
        # count is tracked EXPLICITLY (Req.spec_mapped_tail): plain decode's
        # device_len = cached_len + 1 is the pending-token slot whose table
        # row is stale, NOT an allocation — inferring the tail from
        # device_len > cached_len double-freed a live page on every plain
        # request (the 8195 != 8194 integrity crash on the non-spec server).
        if req.spec_mapped_tail:
            tail = self.engine.page_table[req.table_idx, req.cached_len : req.cached_len + req.spec_mapped_tail]
            # 0 = the unmapped sentinel (a reject zeroed row B's entry);
            # freeing it would hand out a phantom page.
            self.cache_manager._free(tail[tail != 0])
            self.engine.page_table[req.table_idx, req.cached_len : req.cached_len + req.spec_mapped_tail] = 0
            req.spec_mapped_tail = 0
        # Polymorphic free: the DSV4 manager returns the request's window pages + cmp/idx blocks
        # to their tier free-lists; the generic manager frees its KV pages (it reads
        # page_table[req.table_idx], so free the table entry after).
        self.cache_manager.cache_req(req, finished=True)
        self.table_manager.free(req.table_idx)
        req.table_idx = -1

    def _reply_rebuild(self, request_id: str, status: str, error: str | None = None) -> None:
        # Single source of truth with the rollback snapshot (_current_cache_geometry): mamba is
        # usable slots (padding sink excluded, matching the status-bar gauge), and num_swa_pages
        # reports 0 unless the model actually has a window pool.
        geo = self._current_cache_geometry()
        self.send_result(
            [
                CacheRebuildResultMsg(
                    request_id=request_id,
                    status=status,
                    moe_cache_size=geo["moe_cache_size"] or 0,
                    num_pages=geo["num_pages"],
                    mamba_slots=geo["num_mamba_slots"] or 0,
                    num_swa_pages=geo["num_swa_pages"] or 0,
                    error=error,
                )
            ]
        )

    def _execute_pending_rebuild(self) -> None:
        from freetoken.engine.engine import CacheRebuildRejected

        msg = self._pending_rebuild
        assert msg is not None
        self._pending_rebuild = None
        requested = {
            "moe_cache_size": msg.moe_cache_size,
            "num_pages": msg.num_pages,
            "num_mamba_slots": msg.num_mamba_slots,
            "num_swa_pages": msg.num_swa_pages,
        }
        # Rollback target: the CURRENT (serving) sizes of ONLY the pools this request touches.
        # Passing the untouched pools too would trip rebuild_cache's KV/mamba/SWA gate and wipe
        # the prefix cache that a successful resize of just the requested pool preserves.
        snapshot = self._current_cache_geometry()
        prior = {k: snapshot[k] for k, v in requested.items() if v is not None}
        # Cleared here, set by engine.rebuild_runtime_cache at its point of no return — lets the
        # except below tell a pre-teardown failure (engine untouched) from a mid-teardown one.
        self.engine.rebuild_teardown_started = False
        try:
            self.rebuild_cache(**requested)
        except CacheRebuildRejected as e:
            # Rejected before any destructive free — old cache intact, keep serving.
            logger.warning(f"cache rebuild rejected: {e}")
            self._reply_rebuild(msg.request_id, "rejected", error=str(e))
            return
        except Exception as e:  # noqa: BLE001
            if not getattr(self.engine, "rebuild_teardown_started", True):
                # Failed before the destructive phase began: graphs and pools are untouched and
                # the engine is still serving. A destructive rollback would only add risk.
                logger.error(f"cache rebuild failed before teardown: {e!r} — old cache intact")
                self._reply_rebuild(msg.request_id, "rejected", error=repr(e))
                return
            if self.config.tp_info.size > 1:
                # A lone-rank failure cannot be rolled back symmetrically: rebuild_cache runs TP
                # barriers, and ranks that succeeded will not re-enter them — a solo rollback
                # would desync the group. Keep the latch-failed behavior for tp>1.
                logger.error(f"cache rebuild failed: {e!r} — tp>1, latching failed")
                self._reply_rebuild(msg.request_id, "failed", error=repr(e))
                return
            # The destructive phase failed — typically a CUDA OOM while reallocating a pool or
            # recapturing graphs. The graphs/pools are already torn down, so the engine cannot
            # serve as-is. Rather than latch "failed" (which forces a full process restart),
            # rebuild the touched pools back to the sizes that were serving a moment ago: they
            # fit before, so shrinking back frees the just-attempted allocation and restores
            # service. Only if the rollback ALSO fails is the engine genuinely wedged. (Post-OOM
            # CUDA state is not guaranteed sane — a rollback that succeeds here may still surface
            # a deferred fault on a later request; that residual risk is accepted over always
            # forcing a restart.)
            logger.error(f"cache rebuild failed: {e!r} — rolling back to the previous geometry")
            try:
                self.rebuild_cache(**prior)
            except Exception as e2:  # noqa: BLE001 — rollback failed too; genuinely unrecoverable
                logger.error(f"cache rebuild rollback failed: {e2!r} — server latched failed")
                self._reply_rebuild(
                    msg.request_id,
                    "failed",
                    error=f"{e!r}; rollback to the prior geometry also failed: {e2!r}",
                )
                return
            logger.warning("cache rebuild rolled back to the previous geometry — still serving")
            self._log_cache_geometry("Cache rolled back")
            self._reply_rebuild(
                msg.request_id, "rejected", error=f"rebuild failed and was rolled back: {e!r}"
            )
            return
        # Outside the try: an ack/send failure after a fully-applied rebuild must not be
        # mistaken for a rebuild failure and roll back the geometry the engine now serves.
        self._log_cache_geometry("Cache rebuilt")
        self._reply_rebuild(msg.request_id, "ok")

    def _current_cache_geometry(self) -> dict:
        """The pools' current (serving) sizes as rebuild_cache kwargs — the rollback snapshot and
        the single source for _reply_rebuild's readout. None for a pool this model lacks
        (rebuild_cache skips those; the reply maps them to the wire format's 0). num_swa_pages is
        the CONCRETE current window (usable pages) so a rollback restores it byte-for-byte,
        whether it was pinned or ratio-derived."""
        eng = self.engine
        config = self.config
        mc = config.model_config
        num_swa_pages = None
        if getattr(mc, "dsv4_args", None) is not None:
            sizes = getattr(eng.kv_cache, "sizes", None)
            if sizes is not None:  # usable window pages = physical n_win_pages minus the dummy page
                num_swa_pages = max(0, sizes.n_win_pages - 1)
        elif getattr(mc, "has_swa_attention", False) and (
            getattr(config, "cache_type", None) == "swa_radix"
        ):  # usable window tokens = pool tokens minus the slot-0 sentinel
            num_swa_pages = max(0, int(getattr(eng.kv_cache, "swa_num_tokens", 0) or 0) - 1)
        return dict(
            num_pages=eng.num_pages,
            moe_cache_size=eng.moe_offload_cache.cache_size if eng.moe_offload_cache is not None else None,
            num_mamba_slots=(eng.linear_state_pool.num_slots - 1) if eng.linear_state_pool is not None else None,
            num_swa_pages=num_swa_pages,
        )

    def _log_cache_geometry(self, event: str) -> None:
        """One-line readout of every pool's new size + VRAM after a rebuild changed them:
        full KV always; swa/mamba/MoE only for models with the pool. Byte figures are
        best-effort (0 when a unit cost cannot be measured) and must never block the reply."""
        from freetoken.kvcache.cache_status import compute_cache_pools, compute_cache_unit_bytes

        try:
            pools = compute_cache_pools(self.engine)
            unit = compute_cache_unit_bytes(self.engine)
            kv_tokens = pools["num_pages"] * pools["page_size"]
            parts = [
                f"KV {pools['num_pages']} pages"
                f" ({kv_tokens} tokens, {_gib(kv_tokens * unit['kv_bytes_per_token'])})"
            ]
            if pools["num_swa_pages"]:
                swa_tokens = pools["num_swa_pages"] * pools["swa_page_size"]
                parts.append(
                    f"swa {pools['num_swa_pages']} pages"
                    f" ({swa_tokens} tokens, {_gib(swa_tokens * unit['swa_bytes_per_token'])})"
                )
            if pools["num_mamba_slots"]:
                parts.append(
                    f"mamba {pools['num_mamba_slots']} slots"
                    f" ({_gib(pools['num_mamba_slots'] * unit['mamba_bytes_per_slot'])})"
                )
            moe = self.engine.moe_offload_cache
            if moe is not None:
                parts.append(
                    f"MoE cache {moe.cache_size}/{moe.num_layers * moe.num_experts}"
                    f" ({_gib(moe.cache_size * unit['moe_bytes_per_expert'])})"
                )
            logger.info_rank0(f"{event}: " + ", ".join(parts))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"could not log cache geometry: {e!r}")

    def _prepare_batch(self, batch: Batch) -> ForwardInput:
        self.engine.graph_runner.pad_batch(batch)
        self._forward_iter += 1
        if batch.is_decode:
            # Spec-to-plain bridge: a req leaving the spec loop (accept
            # commits cached_len = device_len, extend_len 0 — the staged
            # next input rides on req.spec_next_input) must re-enter the
            # plain decode protocol, which requires EXACTLY one
            # unprocessed position [cached_len, device_len) per req (the
            # GDN/conv decode kernel asserts one token row per request;
            # an extend_len-0 req contributes no hidden row and the
            # cu_seqlens-vs-cache_indices lengths diverge — the
            # "conv_state_indices must have shape (batch_size)" crash).
            # Bridge: stage the resolved next input as this iteration's
            # extend token — token_pool (the plain gather's source), the
            # host ids buf (stop-string mirror), and device_len = cached + 1.
            for req in batch.reqs:
                if req.extend_len == 0 and req.spec_next_input is not None:
                    tok = req.spec_next_input
                    self.token_pool[req.table_idx, req.cached_len] = tok
                    req.input_ids = req._ids_buf[: req.cached_len + 1]
                    req.input_ids[req.cached_len] = tok
                    req.device_len = req.cached_len + 1
            # Free each decoding request's now-out-of-window SWA slots BEFORE the alloc below,
            # so they can back the new token -- this is what bounds the per-request swa
            # footprint during decode. (no-op unless the model is SWA / paged swa pool.)
            self.cache_manager.maybe_free_swa_out_of_window(
                batch.reqs, forward_iter=self._forward_iter)
            for req in batch.reqs:
                req.decode_batch_idx += 1
            # Spec-armed batches skip the plain alloc: _forward dispatches to
            # _forward_spec, whose _prepare_spec_batch advances device_len by 2
            # and allocates the WHOLE span [cached_len, device_len). Allocating
            # the pending token's page here too would double-map position
            # cached_len (the spec alloc overwrites the entry; the first page
            # leaks — 1 page per spec iteration, the 8159 != 8194 drift).
            if not self._spec_armed(batch):
                self.cache_manager.allocate_paged(batch.reqs)
        else:
            # Prefill sibling of the decode driver: free out-of-window swa BEFORE allocating
            # this chunk, so a chunked prompt longer than the swa pool never accumulates its
            # whole swa footprint (which would exhaust alloc_swa). No-op unless SWA/paged.
            self.cache_manager.free_swa_out_of_window_extend(batch.reqs)
            self.cache_manager.allocate_paged(batch.reqs)
        if batch.is_prefill:
            self._gather_multimodal(batch)
        batch.positions = _make_positions(batch, self.device)
        input_mapping = _make_input_tuple(batch, self.device)
        write_mapping = _make_write_tuple(batch, self.device)
        batch.out_loc = self.engine.page_table[input_mapping]
        if self.engine.linear_state_pool is not None:
            if batch.is_decode:
                # GPU GDN-state slot (one per padded request) for the decode gather/scatter;
                # lands in the CUDA-graph input buffer via copy_from. Gate on the cache mode,
                # NOT on whether any padded req has a linear_slot_idx -- the persistent dummy
                # req always carries one (= padding_slot), so that test is True even for naive
                # and would collapse all real naive reqs onto the padding slot. Hybrid: build
                # per padded req from Req.linear_slot_idx (dummy -> padding_slot). Naive: keep
                # the old keying = input_mapping's table_idx column (already staged, no H2D).
                if self.cache_manager.is_hybrid:
                    pool = self.engine.linear_state_pool
                    slots = [r.linear_slot_idx if r.linear_slot_idx is not None
                             else pool.padding_slot for r in batch.padded_reqs]
                    batch.linear_table_idx = torch.tensor(
                        slots, dtype=torch.int32, device="cpu", pin_memory=True
                    ).to(self.device, non_blocking=True)
                else:
                    batch.linear_table_idx = input_mapping[0].to(torch.int32)
            # Per-forward GDN metadata (cu_seqlens / cache_indices / continuation flags),
            # built once here instead of rebuilt in each of the 30 GDN layers. For decode
            # under CUDA graph the persistent cu_seqlens buffer is supplied by set_batch.
            batch.fla_metadata = build_fla_metadata(batch, self.device)
        if batch.is_decode:
            # This batch's padded per-row page-table rows. Backends that snapshot the table for
            # a captured replay (DSV4) read them in prepare_metadata / prepare_for_replay.
            batch.active_table_idx = input_mapping[0].view(-1)
        self.engine.attn_backend.prepare_metadata(batch)
        return ForwardInput(
            batch=batch,
            sample_args=self.engine.sampler.prepare(batch),
            input_tuple=input_mapping,
            write_tuple=write_mapping,
        )

    def _gather_multimodal(self, batch: Batch) -> None:
        """Concatenate per-request vision soft tokens (in request order) for a prefill
        batch so the model can scatter them at image-token positions. ``req.mm_embeds``
        is kept (not cleared) so the cache manager can recognize multimodal requests and
        keep them out of the shared prefix cache (image placeholders share a token id but
        carry per-image content)."""
        parts = [req.mm_embeds for req in batch.reqs if req.mm_embeds is not None]
        if parts:
            batch.mm_embeds = torch.cat(parts, dim=0)

    def _schedule_next_batch(self) -> ForwardInput | None:
        # TODO: support other policies: e.g. DECODE first
        batch = (
            self.prefill_manager.schedule_next_batch(self.prefill_budget)
            or self.decode_manager.schedule_next_batch()
        )
        if batch is None:
            return None
        forward_input = self._prepare_batch(batch)
        self._report_prompt_admissions(batch)
        return forward_input

    def _report_prompt_admissions(self, batch: Batch) -> None:
        """Publish first-prefill accounting only after batch preparation succeeded.

        ``send_result`` is rank-aware: TP rank 0 forwards the signal, other ranks are
        no-ops. The offline handler explicitly ignores this online-accounting message.
        """
        if not batch.is_prefill or not batch.prompt_admissions:
            return
        self.send_result(
            [
                PromptAdmittedMsg(uid=uid, prompt_tokens=prompt_tokens, cached_tokens=cached_tokens)
                for uid, prompt_tokens, cached_tokens in batch.prompt_admissions
            ]
        )

    def _flush_abort_acks(self) -> None:
        pending = getattr(self, "_pending_abort_acks", None)
        if not pending:
            return
        uids = sorted(pending)
        pending.clear()
        self.send_result([ErrorReplyMsg(uid=uid, error="request aborted") for uid in uids])

    def _forward(self, forward_input: ForwardInput) -> ForwardOutput:
        batch, sample_args, input_mapping, output_mapping = forward_input
        batch.input_ids = self.token_pool[input_mapping]
        if self.toolcall_anchor_id is not None and not batch.is_prefill:
            self.cache_manager.snapshot_toolcall_anchor(batch.reqs)

        if self._spec_armed(batch):
            return self._forward_spec(batch, output_mapping)

        forward_output = self.engine.forward_batch(batch, sample_args)
        self.token_pool[output_mapping] = forward_output.next_tokens_gpu
        self.decode_manager.filter_reqs(forward_input.batch.reqs)
        return forward_output

    # ------------------------------------------------------------------
    # MTP spec decode (wave 2 Stage 1): the decode step's spec arm.
    # ------------------------------------------------------------------

    def _spec_armed(self, batch: Batch) -> bool:
        """Stage-1 arm gate. The verify runs as 2 SEQUENTIAL 1-row batches
        per req (row A then row B), so each req's GDN state is consumed in
        stream order — trivially correct ordering that Stage 3 (verify
        graph batching) must re-prove when it fuses the rows. Requires
        page_size == 1 (the reject path frees a mid-page slot otherwise);
        every req must hold a pending carry (None = degrade to plain).
        """
        eng = self.engine
        return (
            eng.mtp_drafter is not None
            and batch.is_decode
            and eng.ctx.page_size == 1
            and all(r.sampling_params.is_greedy and r.spec_carry is not None
                    # room for 2 more positions: the +2 advance writes row A
                    # (certain) and row B (speculative) into the req's ids buf,
                    # which is sized to the exact output budget (max_device_len
                    # = prompt + max_tokens). A req near budget exhaustion
                    # degrades to plain decode for its last tokens — the ids
                    # buf would overflow otherwise (IndexError at the row-B
                    # host write, core.py:94 assert at the drain append).
                    and (r.max_device_len - r.device_len) >= 2
                    for r in batch.reqs)
        )

    def _forward_spec(self, batch: Batch, output_mapping: Indice2D) -> ForwardOutput:
        """One spec decode iteration: draft -> 2-row trunk verify ->
        resolve -> rollback (page slots + GDN state), entirely inside the
        scheduler's _forward so every req mutation completes before the
        overlap scheduler prepares the next batch.

        The verify batches are built through the EXISTING extend
        machinery: each req's cached_len/device_len are advanced 2 slots
        ahead by _prepare_spec_batch (the mechanism prefill extends
        already use — device_len leads cached_len; extend_len = the
        unprocessed span; allocate_paged allocates the whole span; the
        attention/GDN kernels process [cached_len, device_len)). Each
        verify row batch is a phase="prefill" Batch with 1 row/req;
        the trunk forward, the GDN chunk kernel, and the attention extend
        path run per row group. The engine's _forward_spec_batch then
        resolves + runs the MTP replay; this fn rolls rejects back.
        """
        eng = self.engine
        drafter = eng.mtp_drafter
        self._prepare_spec_batch(batch)
        # The GDN snapshot is taken by the engine BETWEEN the row-A and
        # row-B forwards (mid-verify: the state after row A = position q —
        # matching the reject's committed frontier [0, q+1)). The old
        # pre-verify snapshot was one row stale vs cached_len and lagged
        # the attention state under rejects (the repetition-loop bug).
        forward_output = eng._forward_spec_batch(batch)
        self.status_reporter.report_spec_window(
            batch.size, sum(1 for r in batch.reqs if r.spec_accepted))
        self.token_pool[output_mapping] = forward_output.next_tokens_gpu
        # Rollback the rejects NOW (before filter_reqs / the next batch's
        # prepare): free row B's page slots, restore the MID-VERIFY GDN
        # snapshot (state after row A), rewind device_len (spec_undone
        # marks the undone position for the next iteration's row A).
        self._rollback_spec_rejects(batch)
        self.decode_manager.filter_reqs(batch.reqs)
        import os as _os
        if _os.environ.get("FT_SPEC_TRACE"):
            import sys as _sys
            for req in batch.reqs:
                _sys.stderr.write(
                    f"[speciter] uid={req.uid} acc={req.spec_accepted} "
                    f"cached={req.cached_len} dev={req.device_len} "
                    f"free={len(self.cache_manager.free_slots)}\n")
        return forward_output

    def _rollback_spec_rejects(self, batch: Batch) -> None:
        """Per-req post-resolve bookkeeping for the spec iteration (runs
        inside _forward, before filter_reqs — every req mutation completes
        before the overlap scheduler prepares the next batch).

        Full accept (k == depth n): cached_len = device_len — the verify
        committed ALL rows (every position verified; the KV content is
        real), so allocate_paged's next span [cached_len, device_len) is
        exactly the next iteration's new positions.

        k < n: free the UNVERIFIED draft rows' page slots (positions
        q+k+1..q+n — the next iteration's verify re-writes them with
        deterministic content, so freeing is correct), restore the
        MID-VERIFY GDN snapshot taken after verify row k (the state
        covering exactly the committed frontier [0, q+k+1)), rewind
        device_len to q+k+1, and record spec_undone = n - k. cached_len
        ADVANCES to the rewound device_len (rows 0..k are processed AND
        verified — the committed span never shrinks under consecutive
        rejects).
        """
        cm = self.cache_manager
        pool = self.engine.linear_state_pool
        snapshot_slot_lists = batch.spec_gdn_snapshot_slots
        n = self.engine.config.spec_draft_n
        with cm.lazy_free_region():
            for i, req in enumerate(batch.reqs):
                k = req.spec_accept_count
                if k >= n:
                    req.cached_len = req.device_len
                else:
                    # 1. free the unverified draft rows' page slots
                    # [q+k+1, q+n); the page-table entries are zeroed so
                    # the next allocate_paged re-maps them.
                    head, tail = req.cached_len + k + 1, req.device_len
                    cm._free(self.engine.page_table[req.table_idx, head:tail])
                    self.engine.page_table[req.table_idx, head:tail] = 0
                    # 2. restore the GDN snapshot taken after verify row k
                    # into the live slot (a reject at k needs exactly this).
                    if snapshot_slot_lists:
                        pool.copy_from(snapshot_slot_lists[i][k],
                                       req.linear_slot_idx)
                    # 3. rewind device_len to the committed frontier
                    # q+k+1; cached_len advances to match (docstring).
                    req.device_len = head
                    req.cached_len = head
                req.spec_undone = n - k
                req.spec_mapped_tail = 0
        # the extra per-iteration snapshot slots are batch scratch:
        # release them for the whole batch regardless of per-req outcome
        if batch.spec_gdn_extra_slots and pool is not None:
            pool.free(batch.spec_gdn_extra_slots)
            batch.spec_gdn_extra_slots = None

    def _prepare_spec_batch(self, batch: Batch) -> None:
        """Advance each req by n+1 device positions, allocate the fresh
        page slots (the existing allocate_paged path), chain the MTP
        draft n times, build the n+1 1-row verify batches (see
        _make_spec_row_batch) and stage the MTP replay's tensors.
        Consumes the PREVIOUS iteration's resolve state off the reqs
        (spec_carry / spec_next_input).

        The scheduler advances device_len by n+1 (extending the req's
        live region by the verify rows) BEFORE the forwards, and
        allocate_paged covers [cached_len, device_len). Row 0's token is
        the req's certain input (c, or the reject's corrected argmax
        staged in spec_next_input); draft rows 1..n carry the chained
        drafts d_0..d_{n-1}.
        """
        eng = self.engine
        device = self.device
        n = eng.config.spec_draft_n
        rows = n + 1
        # --- advance: n+1 device positions per req, writing the tokens ---
        for req in batch.reqs:
            # Row 0 token: the next certain input (after a reject this is
            # the corrected row-0 argmax the previous resolve staged).
            # The verify rows are [cached_len, cached_len + n + 1): row 0
            # writes AT cached_len — the pending token lives at
            # [cached_len] (plain protocol: device_len = cached_len + 1
            # with the token staged there); writing it at the advanced
            # device_len would duplicate the token and leave an
            # allocated-but-never-written slot inside the window.
            req.input_ids = req._ids_buf[: req.cached_len + rows]
            req.input_ids[req.cached_len] = req.spec_next_input
            req.device_len = req.cached_len + rows
        cm = self.cache_manager
        cm.allocate_paged(batch.reqs)  # covers [cached_len, device_len)
        # From here to the resolve, the verify pages are mapped-but-
        # uncommitted: an abort/finish in this window frees them via
        # spec_mapped_tail (the plain decode's stale-table-row tail must
        # NOT be inferred from device_len > cached_len — see
        # Req.spec_mapped_tail).
        for req in batch.reqs:
            req.spec_mapped_tail = req.device_len - req.cached_len
        self._forward_iter += 1
        cm.maybe_free_swa_out_of_window(batch.reqs, forward_iter=self._forward_iter)
        # --- draft chain + carry staging ---
        # Per-req carries mix devices: prefill seeds spec_carry as a GPU
        # hidden row, every resolve refreshes it to CPU (the drain's
        # host-visible staging) — a batch combining fresh admissions with
        # already-speculating reqs stacks both. Normalize per req.
        carries = torch.stack(
            [r.spec_carry.to(device, non_blocking=True) for r in batch.reqs],
            dim=0)
        input_tokens = torch.tensor(
            [req.input_ids[req.cached_len].item() for req in batch.reqs],
            dtype=torch.int32, device=device)
        batch.spec_carry_gpu = carries
        batch.spec_input_tokens_gpu = input_tokens
        # --- the drafts: step k+1 consumes (carry_k, d_k); the chain is
        # greedy argmax all the way down. Each step's host-buf write
        # (d_k at q+1+k) must land BEFORE the row batches are built —
        # the draft rows' token gathers read [q+1..q+n] off the host buf
        # (_row_token_ids); building them earlier gathers the ids_buf's
        # uninitialized memory.
        mtp = eng.model.model.mtp
        # Step 0 of the chain rides row-0 staging; steps 1..n-1 stage at
        # their own rows (built in the loop below): every chain step must
        # write its layer-(num_layers) KV row at the slot its content
        # belongs to, so the verify's replay overwrites it with the
        # identical content instead of garbage.
        draft_batch = self._make_spec_row_batch(batch.reqs, row_idx=0)
        draft_batch.input_ids = input_tokens
        # Wave-2 stage 2: the draft rides the per-bs CUDA-graph family
        # when the engine captured it; eager stays the reference oracle
        # (FT_SPEC_DRAFT_EAGER keeps the runner None) and the fallback
        # for a bs over the family. draft_batch supplies the staged
        # positions/out_loc/metadata both arms consume.
        runner = getattr(eng, "draft_graph_runner", None)
        use_graph = runner is not None and runner.can_draft(batch.size)
        draft_steps: list[torch.Tensor] = []
        tokens = input_tokens
        for k in range(n):
            # Step k stages at ITS row (position q+k, out_loc pt[q+k]):
            # the post-verify replay rewrites that row with the identical
            # content f(d_{k-1} @ q+k, H_{k-1}). Piling every step onto
            # row 0 (the depth-1 layout) let step 1 clobber pt[q] with
            # f(d_0 @ q, H_q) — wrong token, wrong RoPE — and the replay's
            # row-0 attention then read that garbage self-row and commit
            # a corrupt layer-40 history that the NEXT iteration's draft
            # attends: depth-2 acceptance collapse (stage-4 gate,
            # 2026-09-21) with output staying lossless (the trunk never
            # reads layer 40).
            step_batch = (draft_batch if k == 0
                          else self._make_spec_row_batch(batch.reqs, row_idx=k))
            step_batch.input_ids = tokens
            if use_graph:
                with torch.cuda.stream(eng.stream):
                    d, carries = runner.draft(carries, tokens, step_batch)
            else:
                with torch.cuda.stream(eng.stream):
                    with eng.ctx.forward_batch(step_batch):
                        carries, logits = mtp.draft_step(carries, tokens)
                        d = logits.argmax(dim=-1).to(torch.int32)
            draft_steps.append(d)
            if os.environ.get("FT_DEBUG_STORE"):
                import sys as _sys
                _sys.stderr.write(
                    f"[chain k={k}] tok={int(tokens[0].item())} "
                    f"carry={float(carries[0].float().norm()):.4f} "
                    f"d={[int(x) for x in d.tolist()]}\n")
                _sys.stderr.flush()
            tokens = d
        batch.spec_drafts_gpu = torch.stack(draft_steps, dim=1)  # [B, n]
        for i, req in enumerate(batch.reqs):
            for k, d_k in enumerate(draft_steps):
                # the verify's draft row k consumes d_k at position q+1+k;
                # the host ids tail must hold it (the KV position and the
                # ids tail must agree for stop-string matching and radix
                # commits). On reject the rewound view excludes these
                # writes — the next iteration's row-0 write overwrites
                # them with the corrected token.
                req.input_ids[req.cached_len + 1 + k] = int(d_k[i].item())
        # --- MTP replay tensors: rows per req = n+1, positions [q, q+n] ---
        pos_host = torch.empty(rows * batch.size, dtype=torch.int32,
                               pin_memory=True)
        loc_host = torch.empty(rows * batch.size, dtype=torch.int64,
                               pin_memory=True)
        tok_host = torch.empty(rows * batch.size, dtype=torch.int32,
                               pin_memory=True)
        for i, req in enumerate(batch.reqs):
            q = req.cached_len
            pt = self.engine.page_table[req.table_idx]
            for j in range(rows):
                pos_host[rows * i + j] = q + j
                loc_host[rows * i + j] = int(pt[q + j].item())
                tok_host[rows * i + j] = req.input_ids[q + j].item()
        batch.spec_replay_positions = pos_host.to(device, non_blocking=True)
        batch.spec_replay_out_loc = loc_host.to(device, non_blocking=True)
        batch.spec_replay_input_ids = tok_host.to(device, non_blocking=True)
        # --- verify row batches: built AFTER the drafts' host-buf writes
        # so the draft rows' gathers see the real draft tokens ---
        batch.spec_row_batches = [
            self._make_spec_row_batch(batch.reqs, row_idx=k)
            for k in range(rows)]
        # replay metadata: an (n+1)-row extend per req (span [q, q+n]).
        # _SpecRowPair views give each req the (n+1)-token span so the
        # backend builds the (n+1)·bs-row extend metadata; the result
        # rides the batch (spec_replay_attn_metadata) for the engine's
        # replay.
        replay_reqs = [_SpecRowPair(req) for req in batch.reqs]
        replay_meta = Batch(reqs=replay_reqs, phase="prefill")
        replay_meta.padded_reqs = replay_reqs
        eng.attn_backend.prepare_metadata(replay_meta)
        batch.spec_replay_attn_metadata = replay_meta.attn_metadata
        # --- GDN snapshot slot staging (hybrid only): one slot per
        # snapshot row 0..n-1 — slot 0 is the req's idle ping-pong track,
        # slots 1..n-1 are fresh per-iteration pool allocs (released by
        # _rollback_spec_rejects after the resolve).
        if eng.linear_state_pool is not None and cm.is_hybrid:
            pool = eng.linear_state_pool
            per_req_extra: list[list[int]] = [[] for _ in batch.reqs]
            if n > 1:
                extra = pool.alloc((n - 1) * batch.size)
                batch.spec_gdn_extra_slots = extra
                per_req_extra = [
                    extra[i * (n - 1):(i + 1) * (n - 1)]
                    for i in range(batch.size)]
            batch.spec_gdn_snapshot_slots = [
                [r.mamba_ping_pong[r.mamba_next_track_idx]] + per_req_extra[i]
                for i, r in enumerate(batch.reqs)]

    def _make_spec_row_batch(self, reqs: List[Req], *, row_idx: int) -> Batch:
        """A 1-row-per-req batch for one verify row (row_idx 0 = the
        certain row, 1..n = the draft rows): phase="decode", extend_len 1
        per req. cached_len/device_len are NOT touched (the per-row
        batch's own bookkeeping is derived); the row's token ids come
        straight from the staged ids tail ([c, d_0..d_{n-1}] per req:
        row k takes index k) — the token_pool slots at the verify
        positions hold NO valid tokens for this iteration (the output
        write puts one token per iteration at the frontier only).
        Positions/out_loc/metadata use the standard paths on the wrapper
        view (NO allocation here — the batch-level allocate_paged already
        covers all row spans; allocating here would double-allocate and
        clobber the mapped table entries)."""
        # Verify rows are phase="decode" batches, NOT prefill extends: the
        # GDN fla CHUNK kernel's 1-token-chunk-with-initial-state path
        # diverges from the fused decode kernel (proven in-process: with
        # prefill-phase rows the trunk's row-B argmax skipped ahead in the
        # reference sequence — the bonus/accept chain desynced; with
        # decode-phase rows the spec loop is byte-identical to plain greedy
        # decode over 24 steps on the 35B). The decode path handles the
        # 1-token-per-req shape exactly (conv shift-append + fused SSM
        # update), and the attention decode kernel is exact for q_len=1
        # over the full paged window.
        # Each row batch gets its OWN view of the req state: row k
        # processes position q + k (q = the req's cached_len).
        row_reqs = [_SpecRowReq(req, row_idx) for req in reqs]
        b = Batch(reqs=row_reqs, phase="decode")
        b.padded_reqs = row_reqs
        b.positions = _make_positions(b, self.device)
        input_mapping = _make_input_tuple(b, self.device)
        b.out_loc = self.engine.page_table[input_mapping]
        if self.engine.linear_state_pool is not None:
            slots = [r.linear_slot_idx for r in row_reqs]
            b.linear_table_idx = torch.tensor(
                slots, dtype=torch.int32, device="cpu", pin_memory=True
            ).to(self.device, non_blocking=True)
            b.fla_metadata = build_fla_metadata(b, self.device)
        self.engine.attn_backend.prepare_metadata(b)
        b.input_ids = self._row_token_ids(reqs, row_idx).to(
            self.device, non_blocking=True)
        return b

    def _row_token_ids(self, reqs: List[Req], row_idx: int) -> torch.Tensor:
        """[B] int32 host tensor of the row's input tokens: c (row 0) or
        d_k (row k), matching the staged replay tokens."""
        return torch.tensor(
            [req.input_ids[req.cached_len + row_idx].item() for req in reqs],
            dtype=torch.int32)

    def _mtp_replay_batch_meta(self, batch: Batch) -> Batch:
        """Metadata-only batch for the MTP replay's 2-row/req triton
        extend (positions/out_loc staged separately by the caller)."""
        replay = Batch(reqs=list(batch.reqs), phase="prefill")
        replay.padded_reqs = getattr(batch, "padded_reqs", batch.reqs)
        replay.positions = batch.spec_replay_positions
        replay.out_loc = batch.spec_replay_out_loc
        eng = self.engine
        eng.attn_backend.prepare_metadata(replay)
        return replay


class _SpecRowPair:
    """An (n+1)-row view of a Req for the MTP replay's metadata build: the
    full unprocessed span [cached_len, device_len) = the verify window."""

    def __init__(self, req: Req):
        self._req = req

    @property
    def cached_len(self) -> int:
        return self._req.cached_len

    @property
    def device_len(self) -> int:
        return self._req.device_len

    @property
    def extend_len(self) -> int:
        return self._req.device_len - self._req.cached_len

    @property
    def table_idx(self) -> int:
        return self._req.table_idx


class _SpecRowReq:
    """A 1-row view of a Req for one verify row batch (row 0 = the
    certain row, 1..n the draft rows): the row's extend window is
    [q + row, q + row + 1) where q = the req's cached_len. The row
    batch's bookkeeping (positions, out_loc, attention metadata) is
    derived from these properties without mutating the real req."""

    def __init__(self, req: Req, row: int):
        self._req = req
        self._off = row
        # the row's device_len view: one position past the row's token
        self._dev = req.cached_len + 1 + row

    @property
    def cached_len(self) -> int:
        return self._dev - 1

    @property
    def device_len(self) -> int:
        return self._dev

    @property
    def extend_len(self) -> int:
        return 1

    @property
    def table_idx(self) -> int:
        return self._req.table_idx

    @property
    def linear_slot_idx(self) -> "int | None":
        return self._req.linear_slot_idx

    @property
    def mamba_ping_pong(self):
        return self._req.mamba_ping_pong

    @property
    def mamba_next_track_idx(self) -> int:
        return self._req.mamba_next_track_idx

    @property
    def mamba_last_track_seqlen(self):
        return None  # no ×CHUNK track on a 1-row verify extend


def _make_positions(batch: Batch, device: torch.device) -> torch.Tensor:
    needed_size = sum(r.extend_len for r in batch.padded_reqs)
    indices_host = torch.empty(needed_size, dtype=torch.int32, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        torch.arange(
            req.cached_len,
            req.device_len,
            dtype=torch.int32,
            out=indices_host[offset : offset + length],
        )
        offset += length
    return indices_host.to(device, non_blocking=True)


def _make_input_tuple(batch: Batch, device: torch.device) -> Indice2D:
    mapping_host = torch.empty(len(batch.positions), dtype=torch.int64, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        mapping_host[offset : offset + length].fill_(req.table_idx)
        offset += length
    return mapping_host.to(device, non_blocking=True), batch.positions.to(torch.int64)


def _make_write_tuple(batch: Batch, device: torch.device) -> Indice2D:
    mapping_list = [req.table_idx for req in batch.reqs]
    mapping_host = torch.tensor(mapping_list, dtype=torch.int64, pin_memory=True)
    write_list = [(req.device_len if req.can_decode else -1) for req in batch.reqs]
    write_host = torch.tensor(write_list, dtype=torch.int64, pin_memory=True)
    return mapping_host.to(device, non_blocking=True), write_host.to(device, non_blocking=True)
