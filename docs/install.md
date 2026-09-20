# Install

## Requirements

- Linux x86_64, NVIDIA GPU, driver r580+ (CUDA 13)
- Python >= 3.10, with [uv](https://docs.astral.sh/uv/) recommended (plain
  `pip` + `venv` works too)
- Host RAM sized for the MoE strategy: `offload`/`hybrid` pin expert banks in
  host RAM (~22 GiB for Qwen3.6-35B-A3B-NVFP4; some models add more —
  Qwen3.8-Flash-Next keeps a 47.7 GiB PLE table pinned). See [models.md](models.md).
- CUDA-13-capable NVIDIA GPU. Verified on Blackwell (sm_120). The optional
  vLLM/Marlin W4A16 extra below targets sm_80-99 only.
- Building from source or hitting a JIT fallback needs a CUDA 13 toolkit
  (`nvcc` on PATH) and `ninja` (torch's extension builder requires it).

## Dependency stack

`pyproject.toml` is the contract: ranges run floor = last-verified, ceiling =
next major; there is no lockfile. What each layer is for:

| Layer | Packages | Role |
|---|---|---|
| Runtime core | `torch` 2.11 (cu130 build), `triton` 3.6.0, `transformers` >=5.5, `numpy`, `einops`, `safetensors`, `tqdm` | Tensors, configs, tokenizers, and the Triton attention / GDN / TurboQuant kernels |
| Own native kernels | `apache-tvm-ffi` (pinned), `flashlib` 0.3.0, plus three C++ extensions built at install: `_pinned_tensor`, `_cpu_moe`, `_ple_store` | tvm-ffi–loaded CUDA kernels and the device-side expert-cache LRU (`flashlib`); pinned-memory registration; CPU MoE executor (AVX512-BF16 with runtime dispatch); disk PLE row store (Linux-only) |
| JIT fallbacks | CUDA 13 toolkit + `ninja` | `torch.utils.cpp_extension.load` compiles the GGUF and turbo-KV `.cu` kernels on first use; any AOT-cache miss also falls back to JIT (the nightly `freetoken-kernel-cache` wheel ships the common specs prebuilt). nvcc major must match torch's CUDA major — enforced, override with `FREETOKEN_ALLOW_CUDA_MISMATCH=1` |
| Accelerators (`freetoken[accel]`) | `flashinfer-python[cu13]` (`fi`), `sglang-kernel` 0.4.5 (`sgl`) | The `fi` attention backend (auto-selected on this stack) and NVFP4 expert GEMMs. Without them the runtime falls back to pure-Triton kernels. Never co-install `sgl-kernel` (the old package name) alongside `sglang-kernel` |
| Optional, dedicated env only | `vllm` >=0.14,<0.15 | Marlin W4A16 NVFP4 expert GEMM (sm_80-99). Pins `transformers` 4.x — incompatible with the core requirement, so it is not an installable extra |
| Server & CLI | `fastapi`, `uvicorn`, `pydantic`, `openai`, `pyzmq`, `prompt_toolkit`, `partial-json-parser` | OpenAI- and Anthropic-compatible API server, control daemon, `ft shell`, tool-call parsing |
| Model I/O | `huggingface_hub`, `modelscope`, `gguf` | Checkpoint download from HF or ModelScope; GGUF loading |
## Method 1: Install from PyPI

```bash
uv venv && source .venv/bin/activate
uv pip install "freetoken[accel]"
```

  CUDA kernels JIT-compile on first use: a CUDA 13 toolkit with `nvcc` on PATH
  plus `ninja` (torch's extension builder requires it).

## Method 2: Install from source

```bash
git clone https://github.com/FlashML-org/FreeToken.git && cd FreeToken
uv venv && source .venv/bin/activate
uv pip install -e ".[accel]"
```

## Method 3: Nightly wheels

Every night `main` is built into a wheel pair on the rolling
[`nightly` release](https://github.com/FlashML-org/FreeToken/releases/tag/nightly):
the `freetoken` runtime (CPython 3.12, Linux x86_64) and the matching
`freetoken-kernel-cache` with FreeToken's own CUDA kernels prebuilt.
Install both from the URLs on that release page:

```bash
uv pip install \
  "freetoken[accel] @ https://github.com/FlashML-org/FreeToken/releases/download/nightly/<runtime wheel>" \
  "https://github.com/FlashML-org/FreeToken/releases/download/nightly/<kernel-cache wheel>"
```

Filenames carry a `+g<sha>` build stamp and change every night, and the `nightly`
tag moves with them. Pin a wheel URL, never the tag. A local copy of the tag goes
stale: `git fetch` leaves it alone and `git fetch --tags` refuses to overwrite it;
refresh it with `git fetch --force origin tag nightly`. `engine-linux_x86_64.json`
next to the wheels names the current pair for scripts.

## Verify

```bash
source .venv/bin/activate
ft --version
ft serve --model ~/path/to/Qwen3.6-35B-A3B
curl http://127.0.0.1:1919/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.6-35B-A3B","messages":[{"role":"user","content":"hi"}]}'
```

Then head to [quickstart.md](quickstart.md).
