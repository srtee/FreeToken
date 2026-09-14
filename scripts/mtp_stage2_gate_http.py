"""Gate 2 against a LIVE server: 3 prompts, spec vs plain greedy, byte-identity.

Posts each prompt twice (spec + plain toggled per-request is impossible —
the server boots with --spec-mtp; instead compare against a fresh
non-spec server OR reuse the in-process plain outputs). Here: the spec
half rides the live server; the plain half is produced by the same
server with is_greedy params — spec disarms itself when a req is not
greedy, so we get plain-equivalent rows by sending top_k=50, temperature=1
... no: sampling would differ. The honest server-side A/B: the spec
server ALWAYS uses the spec loop for greedy reqs, so the plain reference
comes from the SAME checkpoint via a second in-process LLM run.

Run:  .venv/bin/python scripts/mtp_stage2_gate_http.py [port]
"""
import json
import sys
import urllib.request

PORT = sys.argv[1] if len(sys.argv) > 1 else "1957"
PROMPTS = [
    "The industrial revolution began in Britain during the late eighteenth century. "
    "Describe the three most important technological innovations of this period "
    "and their effects on urbanization:",
    "Write a Python function that computes the n-th Fibonacci number using "
    "memoization. Include docstring and type hints.",
    "Summarize the causes of the First World War in four paragraphs, covering "
    "alliance systems, militarism, imperialism, and nationalism.",
]


def gen(prompt: str, max_tokens: int = 256) -> str:
    body = json.dumps({
        "model": "spec35",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/completions", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)["choices"][0]["text"]


def main() -> None:
    outs = []
    for i, p in enumerate(PROMPTS):
        outs.append(gen(p))
        print(f"prompt {i}: {len(outs[-1])} chars", flush=True)
    json.dump(outs, open("/tmp/stage2_gate_http_spec.json", "w"))
    print("spec-side outputs written to /tmp/stage2_gate_http_spec.json")


if __name__ == "__main__":
    main()