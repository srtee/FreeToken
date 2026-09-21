"""MTP wave-2 Stage 4 gate: depth-2 draft chain vs plain / depth-1 (offline).

Runs the stage-3 gate worker (byte-for-byte the same harness, depth
parametrized via SPEC_DRAFT_N) in three modes and byte-compares:

  Gate 1 — losslessness at depth 2: spec-d2 output == PLAIN reference
           output, token for token (cap-short trailing allowed: depth 2
           can end up to 2 tokens early at the max_tokens cap).
  Gate 2 — depth-1 regression: spec-d1 == plain (the stage-3 gate,
           re-run against the generalized loop — this is the refactor's
           own regression gate).
  Gate 3 — depth agreement: spec-d2 == spec-d1 (both are lossless, so
           they must agree with each other everywhere too).

Each spec worker also runs the in-process eager-vs-graphed verify-row
bit-equality gate and prints a stats line (depth, gen wall time,
per-position accept counts) that this driver surfaces in the perf table.

Env: CKPT, GATE_TOKENS; run one arm alone: mtp_stage4_gate.py <mode>
(modes: plain | d1 | d2).
Run:  .venv/bin/python scripts/mtp_stage4_gate.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

MODES = (("plain", {}, 2463), ("d1", {"SPEC_DRAFT_N": "1"}, 2464),
         ("d2", {"SPEC_DRAFT_N": "2"}, 2465))


def run_worker(mode: str, extra_env: dict[str, str], port: int) -> dict:
    from mtp_stage3_gate import PROMPTS  # noqa: F401  (import check only)

    script = os.path.join(os.path.dirname(__file__), "mtp_stage3_gate_worker.py")
    env = dict(os.environ, FT_DIST_PORT=str(port), **extra_env)
    out = subprocess.run(
        [sys.executable, "-u", script, "plain" if mode == "plain" else "spec"],
        capture_output=True, text=True, check=False, env=env, timeout=3600)
    if out.returncode != 0:
        print(out.stdout[-4000:])
        print(out.stderr[-4000:], file=sys.stderr)
        raise RuntimeError(f"worker '{mode}' failed rc={out.returncode}")
    lines = out.stdout.strip().splitlines()
    stats = json.loads(lines[-2])
    return {"tokens": json.loads(lines[-1]), "stats": stats}


def compare(tag: str, ref: list, spec: list, short: int, long: int = 0) -> int:
    """Byte-compare; spec arm's length must land in [len(ref)-short,
    len(ref)+long] (cap-trailing and cap-overshoot allowances)."""
    total = 0
    for i, (p, s) in enumerate(zip(ref, spec)):
        n = min(len(p), len(s))
        diffs = [j for j in range(n) if p[j] != s[j]]
        if diffs:
            raise AssertionError(
                f"{tag} prompt {i}: token mismatch at {diffs[:4]} of {n}")
        if not len(p) - short <= len(s) <= len(p) + long:
            raise AssertionError(
                f"{tag} prompt {i}: length {len(p)} vs {len(s)}")
        total += n
    print(f"{tag}: byte-identical on all prompts ({total} tokens compared)")
    return total


def main() -> None:
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        cfg = dict(MODES)[mode]
        out = run_worker(mode, cfg, dict((m, p) for m, _, p in MODES)[mode])
        print(json.dumps(out["tokens"][-1][:16]))
        return

    outs = {}
    for mode, extra_env, port in MODES:
        print(f"--- worker '{mode}' starting ---", flush=True)
        outs[mode] = run_worker(mode, extra_env, port)

    plain = outs["plain"]["tokens"]
    compare("GATE 2 (d1 == plain)", plain, outs["d1"]["tokens"], short=1)
    compare("GATE 1 (d2 == plain)", plain, outs["d2"]["tokens"], short=2)
    compare("GATE 3 (d2 == d1)", outs["d1"]["tokens"], outs["d2"]["tokens"],
            short=1, long=1)

    print("\nPerf table (3 prompts x GATE_TOKENS, bs=1, greedy):")
    print(f"{'mode':<8} {'depth':>5} {'secs':>8} {'tokens':>7} {'tok/s':>7} "
          f"{'acc':>7} {'acc@0':>6} {'acc@1':>6}")
    for mode, _, _ in MODES:
        st = outs[mode]["stats"]
        if st is None:
            print(f"{mode:<8} {'-':>5} {'':>8}")
            continue
        rate = st["spec_accepted"] / st["spec_drafted"] if st["spec_drafted"] else 0.0
        acc = st["accepted_at"] + [0] * (2 - len(st["accepted_at"]))
        print(f"{mode:<8} {st['depth']:>5} {st['gen_secs']:>8.2f} "
              f"{st['tokens']:>7} {st['tokens'] / st['gen_secs']:>7.1f} "
              f"{rate:>7.2f} {acc[0]:>6} {acc[1]:>6}")

    d2 = outs["d2"]["stats"]
    rate = d2["spec_accepted"] / d2["spec_drafted"] if d2["spec_drafted"] else 0.0
    print(f"\ndepth-2 acceptance: {rate:.3f} (accepted_at={d2['accepted_at']})")
    print("STAGE 4 GATE: ALL PASS")


if __name__ == "__main__":
    main()
