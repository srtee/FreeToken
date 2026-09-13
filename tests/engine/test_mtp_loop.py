"""MTP spec-loop iteration state machine (wave-2 Stage 0.2).

Pure-logic model of the corrected loop (docs/mtp-wave2-plan.md; semantics
frozen against buun in docs/mtp-spec-semantics.md). Per iteration with
certain input c@q and carry H (the trunk hidden at the newest verified
position):

  d = draft(H, c)          # 1-row MTP forward; writes layer-40 KV[q]
  rows A,B = verify(c@q, d@(q+1))
  a = argmax(A)            # trunk's true token for q+1  — verification of d
  b = argmax(B)            # bonus candidate for q+2
  accept (a == d):  emit [d, b];  next = b@q+2;  carry = B.hidden
  reject (a != d):  emit [a];     next = a@q+1;  carry = A.hidden;
                    free KV[q+1] (row B's slot), device_len -= 1

The layer-40 invariant: rows exist densely for every position < q.
Every expected value below is hand-computed in comments. A wrong loop
(wrong carry source, missing rollback, wrong next-input selection)
fails loudly.

CPU-only, deterministic, no engine imports beyond the pure-logic
`verify_chain`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from freetoken.engine.spec_mtp import verify_chain


# ---------------------------------------------------------------------------
# The loop model
# ---------------------------------------------------------------------------

class KVPool:
    """Layer-40 paged-KV bookkeeping: slot occupancy per position.

    The toy pool allocates a fresh slot per WRITE (so draft's row at q
    and verify row A's re-write at q are distinct slots); device_len is
    defined as (max position holding a row) + 1, matching the frontier
    semantics of the real pool.
    """

    def __init__(self) -> None:
        self.pos_to_slot: dict[int, int] = {}
        self._next_slot = 0

    def write(self, pos: int) -> int:
        slot = self._next_slot
        self._next_slot += 1
        self.pos_to_slot[pos] = slot
        return slot

    def free(self, pos: int) -> int:
        slot = self.pos_to_slot.pop(pos)
        return slot

    def has(self, pos: int) -> bool:
        return pos in self.pos_to_slot

    def assert_dense(self, frontier: int) -> None:
        """Invariant: a row exists for every position in [0, frontier)."""
        missing = [p for p in range(frontier) if p not in self.pos_to_slot]
        assert not missing, f"layer-40 rows missing at positions {missing}"


@dataclass
class IterationResult:
    emitted: list[int]
    accepted: bool
    next_input: int
    carry_provenance: str    # 'A' or 'B'
    carry_hidden: float
    freed_slot: int | None   # reject: the freed layer-40 slot; accept: None


@dataclass
class LoopState:
    q: int               # position of the NEXT certain input (= frontier)
    input_token: int     # the next certain input token c@q
    carry_provenance: str
    carry_hidden: float
    device_len: int
    kv: KVPool


def draft(carry_hidden: float, kv: KVPool, q: int) -> int:
    """MTP draft step: 1-row forward, writes layer-40 KV[q], returns d.

    Synthetic drafter: d = 1000 + int(carry_hidden) — deterministic, so
    the tables below pin accept/reject per iteration."""
    kv.write(q)
    return 1000 + int(carry_hidden)


def verify(c: int, d: int, kv: KVPool, q: int,
           table: dict[tuple[int, int], tuple[int, int, float, float]],
           ) -> tuple[int, int, float, float]:
    """2-row trunk verify forward for rows [c@q, d@(q+1)].

    Writes layer-40 KV rows: row A at q (re-write of the draft's row —
    same position), row B at q+1. Returns (a, b, hA, hB) from the table:
    a = row-A argmax (verification of d), b = row-B argmax (bonus),
    hA/hB = row hidden states (carry sources under reject/accept)."""
    kv.write(q)      # row A: same position as the draft's row
    kv.write(q + 1)  # row B: the drafted position
    return table[(c, d)]


def resolve(kv: KVPool, q: int, d: int, a: int, b: int,
            hA: float, hB: float) -> IterationResult:
    """Resolve per the semantics doc (accept iff a == d)."""
    if a == d:
        # accept: emit [d, b]; next input b@q+2; carry = row B's hidden
        return IterationResult(
            emitted=[d, b], accepted=True, next_input=b,
            carry_provenance='B', carry_hidden=hB, freed_slot=None)
    # reject: emit [a]; next input a@q+1; carry = row A's hidden; roll
    # back row B (free its KV slot; device_len rewinds by 1)
    freed = kv.free(q + 1)
    return IterationResult(
        emitted=[a], accepted=False, next_input=a,
        carry_provenance='A', carry_hidden=hA, freed_slot=freed)


def run_loop(initial_input: int, initial_carry: float, n_iters: int,
             table: dict[tuple[int, int], tuple[int, int, float, float]],
             ) -> tuple[LoopState, list[IterationResult]]:
    """Run n_iters of the corrected loop; hand-mirrors the semantics doc
    pseudocode. Returns the final state and per-iteration results."""
    q = 0
    c = initial_input
    carry_hidden = initial_carry
    kv = KVPool()
    device_len = 0
    results: list[IterationResult] = []
    for _ in range(n_iters):
        d = draft(carry_hidden, kv, q)
        a, b, hA, hB = verify(c, d, kv, q, table)
        res = resolve(kv, q, d, a, b, hA, hB)
        if res.accepted:
            device_len += 2          # rows q (row A) and q+1 committed
            q += 2
        else:
            device_len += 1          # +2 verify rows written, row B freed
            q += 1
        carry_hidden = res.carry_hidden
        c = res.next_input
        results.append(res)
    state = LoopState(
        q=q, input_token=c, carry_provenance=results[-1].carry_provenance,
        carry_hidden=carry_hidden, device_len=device_len, kv=kv)
    return state, results


# ---------------------------------------------------------------------------
# Synthetic logits tables
# ---------------------------------------------------------------------------
# Each entry maps (c, d) -> (a, b, hA, hB): row-A argmax a (verification),
# row-B argmax b (bonus), row hidden states (carry sources). The drafter
# is d = 1000 + int(carry_hidden), so accept/reject is fully table-pinned.


def make_accept_table(n: int) -> dict:
    """All-accept table for n iterations: row-A argmax == draft each time.

    Hand-computed chain (carry_0 = 5.0, c_0 = 0):
      i:    carry_i   d_i    a_i(=d)   b_i    hA_i  hB_i
      0     5.0       1005   1005      100    1.0   10.0
      1     10.0      1010   1010      101    2.0   20.0
      2     20.0      1020   1020      102    3.0   30.0
      3     30.0      1030   1030      103    4.0   40.0
      4     40.0      1040   1040      104    5.0   50.0
    Next input of iteration i is b_i; next carry is hB_i."""
    t: dict[tuple[int, int], tuple[int, int, float, float]] = {}
    carry = 5.0
    c = 0
    for i in range(n):
        d = 1000 + int(carry)
        t[(c, d)] = (d, 100 + i, float(i + 1), 10.0 * (i + 1))
        c = 100 + i
        carry = 10.0 * (i + 1)
    return t


def make_reject_table(n: int) -> dict:
    """All-reject table for n iterations: row-A argmax = draft + 1.

    Hand-computed chain (carry_0 = 5.0, c_0 = 0; after iteration i the
    next carry is hA_i = i + 2 (row A's hidden) and the next input is
    a_i = d_i + 1, so the next draft is d_{i+1} = 1000 + hA_i):
      i:    carry_i   c_i    d_i    a_i=d+1  hA_i   next carry
      0     5.0       0      1005   1006     2.0    2.0
      1     2.0       1006   1002   1003     3.0    3.0
      2     3.0       1003   1003   1004     4.0    4.0
      3     4.0       1004   1004   1005     5.0    5.0
      4     5.0       1005   1005   1006     6.0    6.0
    """
    t: dict[tuple[int, int], tuple[int, int, float, float]] = {}
    carry = 5.0
    c = 0
    for i in range(n):
        d = 1000 + int(carry)
        t[(c, d)] = (d + 1, 777, float(i + 2), 0.0)
        c = d + 1
        carry = float(i + 2)
    return t

# Mixed pattern table: fixed, hand-written. The drafter is d = 1000 +
# int(carry), so the (c, d) keys chain through the carries:
#   i=0: carry 0.0 -> d=1000; (0, 1000) accept -> next c=2000, carry=hB=2.0
#   i=1: carry 2.0 -> d=1002; (2000, 1002) accept -> next c=4000, carry=hB=4.0
#   i=2: carry 4.0 -> d=1004; (4000, 1004) reject -> next c=999, carry=hA=5.0
#   i=3: carry 5.0 -> d=1005; (999, 1005) reject -> next c=998, carry=7.0
TABLE: dict[tuple[int, int], tuple[int, int, float, float]] = {
    (0, 1000):     (1000, 2000, 1.0, 2.0),   # accept, bonus 2000
    (2000, 1002):  (1002, 4000, 3.0, 4.0),   # accept, bonus 4000
    (4000, 1004):  (999, 5000, 5.0, 6.0),    # reject: a != d
    (999, 1005):   (998, 5000, 7.0, 8.0),    # reject: a != d
}
# Same table with ONE decision flipped: (4000, 1004) now accepts.
TABLE_FLIP: dict[tuple[int, int], tuple[int, int, float, float]] = {
    (0, 1000):     (1000, 2000, 1.0, 2.0),
    (2000, 1002):  (1002, 4000, 3.0, 4.0),
    (4000, 1004):  (1004, 5000, 5.0, 6.0),   # FLIPPED: accept
    (999, 1005):   (998, 5000, 7.0, 8.0),
}

# ---------------------------------------------------------------------------
# (a) all-accept run of 5 iterations
# ---------------------------------------------------------------------------

def test_all_accept_run_of_5():
    n = 5
    table = make_accept_table(n)
    final, results = run_loop(initial_input=0, initial_carry=5.0, n_iters=n,
                              table=table)

    # Hand-computed trace (see make_accept_table's docstring):
    #   i=0: d=1005, a=1005 (accept); emit [1005, 100]; next=100;
    #        carry = hB = 10.0 ('B'); device_len 0 -> 2; q 0 -> 2
    #   i=1: d=1010; emit [1010, 101]; next=101; carry=20.0 ('B'); len 4
    #   i=2: d=1020; emit [1020, 102]; carry=30.0 ('B'); len 6; q 6
    #   i=3: d=1030; emit [1030, 103]; carry=40.0 ('B'); len 8
    #   i=4: d=1040; emit [1040, 104]; carry=50.0 ('B'); len 10; q 10
    expected_emitted = [
        [1005, 100], [1010, 101], [1020, 102], [1030, 103], [1040, 104]]
    assert [r.emitted for r in results] == expected_emitted
    assert all(r.accepted for r in results)
    assert [r.next_input for r in results] == [100, 101, 102, 103, 104]
    # carry provenance: always row B under accept (Q2 of the semantics doc)
    assert [r.carry_provenance for r in results] == ['B'] * 5
    assert [r.carry_hidden for r in results] == [10.0, 20.0, 30.0, 40.0, 50.0]
    assert final.q == 10
    assert final.device_len == 10
    assert final.input_token == 104


def test_all_accept_carry_is_row_B_hidden():
    # Hand-computed: iteration i's carry_hidden == row-B hidden hB_i = 10*(i+1)
    n = 5
    table = make_accept_table(n)
    final, results = run_loop(initial_input=0, initial_carry=5.0, n_iters=n,
                              table=table)
    for i, r in enumerate(results):
        assert r.carry_hidden == 10.0 * (i + 1), (
            f"iteration {i}: carry must be row B's hidden (Q2: accept -> row B)")
    assert final.carry_hidden == 50.0


# ---------------------------------------------------------------------------
# (b) all-reject run
# ---------------------------------------------------------------------------

def test_all_reject_run():
    n = 5
    table = make_reject_table(n)
    final, results = run_loop(initial_input=0, initial_carry=5.0, n_iters=n,
                              table=table)

    # Hand-computed trace (see make_reject_table's docstring):
    #   i=0: q=0, c=0, d=1005, a=1006 (reject); emit [1006]; next=1006;
    #        carry=hA=2.0 ('A'); freed = position 1's slot;
    #        device_len: 0 +2 (verify rows) -1 (row B freed) = 1
    #   i=1: q=1, c=1006, d=1002, a=1003; emit [1003]; carry=3.0; len 2
    #   i=2: q=2, c=1003, d=1003, a=1004; emit [1004]; carry=4.0; len 3
    #   i=3: q=3, c=1004, d=1004, a=1005; emit [1005]; carry=5.0; len 4
    #   i=4: q=4, c=1005, d=1005, a=1006; emit [1006]; carry=6.0; len 5
    assert [r.emitted for r in results] == [[1006], [1003], [1004], [1005], [1006]]
    assert [r.next_input for r in results] == [1006, 1003, 1004, 1005, 1006]
    assert [r.accepted for r in results] == [False] * 5
    # carry provenance: always row A under reject
    assert [r.carry_provenance for r in results] == ['A'] * 5
    assert [r.carry_hidden for r in results] == [2.0, 3.0, 4.0, 5.0, 6.0]
    # rollback bookkeeping: exactly one freed slot per iteration
    assert all(r.freed_slot is not None for r in results)
    assert final.device_len == 5
    assert final.q == 5
    assert final.carry_provenance == 'A'


# ---------------------------------------------------------------------------
# (c) mixed accept/reject pattern from a fixed table
# ---------------------------------------------------------------------------

def test_mixed_pattern_from_fixed_table():
    # Hand trace (see TABLE's comment; carry_0 = 0.0):
    #   i=0: c=0, d=1000; (0,1000) -> a=1000 ACCEPT; emit [1000, 2000];
    #        next=2000; carry=hB=2.0 ('B'); device_len +2 = 2; q 0 -> 2
    #   i=1: c=2000, d=1002; (2000,1002) -> a=1002 ACCEPT; emit [1002, 4000];
    #        next=4000; carry=4.0 ('B'); len 4; q 4
    #   i=2: c=4000, d=1004; (4000,1004) -> a=999 REJECT; emit [999];
    #        next=999; carry=hA=5.0 ('A'); freed slot; len 4+2-1 = 5; q 5
    #   i=3: c=999, d=1005; (999,1005) -> a=998 REJECT; emit [998];
    #        next=998; carry=7.0 ('A'); len 6; q 6
    final, results = run_loop(initial_input=0, initial_carry=0.0, n_iters=4,
                              table=TABLE)
    assert [r.emitted for r in results] == [
        [1000, 2000], [1002, 4000], [999], [998]]
    assert [r.accepted for r in results] == [True, True, False, False]
    assert [r.next_input for r in results] == [2000, 4000, 999, 998]
    # carry: B, B, A, A (Q2: accept -> row B; reject -> row A)
    assert [r.carry_provenance for r in results] == ['B', 'B', 'A', 'A']
    assert [r.carry_hidden for r in results] == [2.0, 4.0, 5.0, 7.0]
    # device_len: accepts add 2; rejects add +2-1 = 1 net
    assert final.device_len == 6
    assert final.q == 6
    # accepts free nothing; rejects free exactly one slot each
    assert [r.freed_slot is None for r in results] == [True, True, False, False]


# ---------------------------------------------------------------------------
# (d) layer-40 KV row bookkeeping (the invariant)
# ---------------------------------------------------------------------------

def test_layer40_invariant_after_accept():
    """After an accept, rows exist for [0, q) — INCLUDING the
    accepted-draft position q+1: row B wrote it and it is never freed
    (semantics doc Q1: no catch-up step is needed)."""
    final, results = run_loop(initial_input=0, initial_carry=5.0, n_iters=1,
                              table=make_accept_table(1))
    assert results[0].accepted
    # q = 2 after one accept: positions 0 and 1 must both have rows
    assert final.q == 2
    assert final.kv.has(0)
    assert final.kv.has(1)
    final.kv.assert_dense(final.q)


def test_layer40_invariant_after_reject():
    """After a reject, row B's row is freed; the frontier rewinds by 1 and
    the invariant (rows < q) still holds — the NEXT iteration's row A
    re-writes the position."""
    final, results = run_loop(initial_input=0, initial_carry=5.0, n_iters=2,
                              table=make_reject_table(2))
    # Hand-computed (toy pool: a fresh slot per write; row B's slot freed):
    #   i=0 @q=0: draft writes pos0, rowA writes pos0, rowB writes pos1,
    #             then pos1 freed -> rows {0}; q 0 -> 1
    #   i=1 @q=1: draft writes pos1, rowA writes pos1, rowB writes pos2,
    #             then pos2 freed -> rows {0, 1}; q 1 -> 2
    # (position 1 IS present at the end: iteration 1's row A re-wrote it.)
    assert final.kv.has(1)
    assert not final.kv.has(2)   # iteration 1's row B was freed
    assert final.q == 2
    final.kv.assert_dense(final.q)


def test_device_len_matches_row_frontier():
    """Invariant: device_len == (max position holding a row) + 1, i.e. the
    layer-40 pool frontier tracks the committed positions."""
    final, _ = run_loop(initial_input=0, initial_carry=0.0, n_iters=4,
                        table=TABLE)
    assert final.device_len == max(final.kv.pos_to_slot) + 1
    # hand-computed for the mixed table: 2, 4, 5, 6 across the 4 iterations
    assert final.device_len == 6


def test_layer40_rows_dense_across_full_mixed_run():
    """The mixed run's final KV must be dense over [0, 6): every position
    the loop advanced over has a row (the invariant the engine must hold
    in the real paged pool)."""
    final, _ = run_loop(initial_input=0, initial_carry=0.0, n_iters=4,
                        table=TABLE)
    final.kv.assert_dense(final.q)


# ---------------------------------------------------------------------------
# (e) verify_chain consistency with the loop model's acceptance mask
# ---------------------------------------------------------------------------

def test_verify_chain_matches_loop_acceptance():
    """The engine's verify_chain must produce the same per-iteration
    accept decisions the loop model derives (a == d at depth 1)."""
    # replay the loop's verify inputs: per-iteration (draft, row-A argmax)
    # pairs along the mixed TABLE trace (hand-computed):
    #   (1000, 1000) accept, (1002, 1002) accept,
    #   (1004, 999) reject,  (1005, 998) reject
    _, results = run_loop(initial_input=0, initial_carry=0.0, n_iters=4,
                          table=TABLE)
    # the pairs are recoverable from the model: d_i = 1000 + int(carry_i),
    # a_i = accepted ? d_i : next_input
    # d: [1000, 1002, 1004, 1005]; a: [1000, 1002, 999, 998]
    drafts = torch.tensor([[1000, 1002, 1004, 1005]], dtype=torch.int64)
    argmaxes = torch.tensor([[1000, 1002, 999, 998]], dtype=torch.int64)
    mask = verify_chain(argmaxes, drafts)
    # hand-computed: accepts at iterations 0,1; rejects at 2,3
    assert mask.tolist() == [[1, 1, 0, 0]]
    assert [r.accepted for r in results] == [bool(mask[0, i]) for i in range(4)]


def test_verify_chain_breaks_chain_on_mid_reject():
    """The cumulative semantics the loop relies on: a chain with a mid
    rejection masks everything after it (depth > 1 behaviour verify_chain
    already guarantees; the depth-1 loop never exercises it, but the
    acceptance MASK contract is what the loop's resolve is built on)."""
    target = torch.tensor([[1000, 999, 555]])
    drafts = torch.tensor([[1000, 888]])
    # row0 argmax 1000 == draft 1000 -> accept; row1 argmax 999 != draft 888
    # -> reject, and the later draft is masked out by the cumprod chain
    assert verify_chain(target, drafts).tolist() == [[1, 0]]


# ---------------------------------------------------------------------------
# (negative control) a flipped accept decision must change the trace
# ---------------------------------------------------------------------------

def test_negative_flip_bites():
    """Negative control: flipping ONE accept decision in the table must
    change the loop's trace — proving the tests bite on a wrong loop.

    TABLE iteration 2 ((4000,1004)) rejects; TABLE_FLIP accepts it.
    Hand-computed divergence at iteration 2:
      TABLE:        i=2 emit [999],        next 999,  carry 'A' (5.0)
      TABLE_FLIP:   i=2 emit [1004, 5000], next 5000, carry 'B' (6.0)
    (TABLE_FLIP's i=3 would hit (5000, 1000+int(6.0)) = (5000, 1006),
    absent from the table — so only 3 iterations are run on the flip.)
    """
    _, base = run_loop(initial_input=0, initial_carry=0.0, n_iters=4,
                       table=TABLE)
    _, flip = run_loop(initial_input=0, initial_carry=0.0, n_iters=3,
                       table=TABLE_FLIP)
    assert base[2].accepted is False
    assert flip[2].accepted is True
    # emissions diverge at the flipped iteration
    assert base[2].emitted != flip[2].emitted
    # carry provenance flips too ('A' vs 'B') and next input differs
    assert base[2].carry_provenance != flip[2].carry_provenance
    assert base[2].next_input != flip[2].next_input