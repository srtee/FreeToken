#!/usr/bin/env python3
"""Logit fingerprint harness for the qwen35moe debug plan.

Two engines, one interface:
  llama --model PATH --prompt TEXT --port N   -> top-k next-token (id, logprob)
  ft     --model PATH --prompt TEXT --port N   -> greedy continuation text

The llama.cpp backend serves the same GGUF coherently and provides
per-token logprobs (n_probs). ft rejects logprobs, so it returns greedy
text only. Cross-engine comparison is text-level for ft (prefix sweep),
logit-level for llama.cpp.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request


def _post(url: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def probe_llama(args) -> dict:
    d = _post(
        f"http://127.0.0.1:{args.port}/completion",
        {"prompt": args.prompt, "n_predict": 1, "temperature": 0,
         "n_probs": args.top_k, "cache_prompt": False},
        timeout=args.timeout)
    content = d.get("content", "")
    probs = d.get("completion_prob", [])[: args.top_k]
    topk = [(p["tok_str"], round(p["logprob"], 4)) for p in probs]
    return {"engine": "llama", "prompt": args.prompt,
            "next_token": content, "top_k": topk}


def probe_ft(args) -> dict:
    d = _post(
        f"http://127.0.0.1:{args.port}/v1/completions",
        {"model": args.served_name, "prompt": args.prompt,
         "max_tokens": args.max_tokens, "temperature": 0},
        timeout=args.timeout)
    text = d["choices"][0]["text"]
    return {"engine": "ft", "prompt": args.prompt,
            "greedy_text": text}


def sweep(args) -> list[dict]:
    """Prefix-length sweep: fixed suffix text, growing prefixes, both engines.

    Reports, per prefix length: ft greedy text vs llama.cpp greedy text.
    Divergence at the FIRST predicted token for some length = prefill bug;
    first token matches but later tokens diverge = decode-state bug.
    """
    words = args.prompt.split(" ")
    rows = []
    for n in range(1, len(words) + 1):
        prefix = " ".join(words[:n])
        try:
            ft = probe_ft(args) if args.ft_port else None
        except Exception as e:
            ft = {"error": str(e)}
        args.prompt = prefix
        try:
            ll = probe_llama(args) if args.llama_port else None
        except Exception as e:
            ll = {"error": str(e)}
        rows.append({"len": len(prefix), "prefix": prefix, "ft": ft, "llama": ll})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--port", type=int, default=8899, help="llama.cpp port")
    ap.add_argument("--ft-port", type=int, default=0, help="ft port (0=off)")
    ap.add_argument("--llama-port", type=int, default=0, help="llama port (0=off)")
    ap.add_argument("--served-name", default="coder")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=180)
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()
    if args.sweep:
        if not (args.ft_port and args.llama_port):
            sys.exit("--sweep needs both --ft-port and --llama-port")
        rows = sweep(args)
        for r in rows:
            ft_txt = r["ft"].get("greedy_text") or r["ft"].get("error", "")
            ll_full = r["llama"].get("next_token", "")
            match = "\u2705" if ft_txt[:1] == ll_full[:1] else "\u274c"
            print(f"[{r['len']:3d}] {match} ft={ft_txt!r:40} llama={ll_full!r}")
    else:
        if args.ft_port:
            print(json.dumps(probe_ft(args)))
        if args.llama_port:
            print(json.dumps(probe_llama(args)))


if __name__ == "__main__":
    main()