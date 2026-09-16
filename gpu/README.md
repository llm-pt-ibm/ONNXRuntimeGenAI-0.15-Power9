# onnxruntime-genai 0.15.0 with CUDA on POWER9

Built against the CUDA build of ONNX Runtime — see
[ONNXRuntime-1.26-Power9/gpu](https://github.com/llm-pt-ibm/ONNXRuntime-1.26-Power9/tree/main/gpu),
which carries the three CUDA patches and the hard-won notes.

Tesla V100 (sm_70), CUDA 12.4.1, cuDNN 9.0.0.312.

## Results

Phi-3-mini-4k int4 (`cuda/cuda-int4-rtn-block-32`):

| | tok/s |
|---|---|
| CPU with the POWER int4 kernel, 80 threads | 10.5 |
| **GPU decode** | **177.8** |
| GPU decode, `enable_cuda_graph=1` | **193.5** |
| GPU prefill | ~208 |

## The patches

**The same four CPU patches apply, unchanged** — they are build plumbing
(architecture detection, the zlib pin, `--ort_home`, patch idempotency) and are
orthogonal to the execution provider. There is no genai-specific CUDA patch:
genai's own CUDA code compiled cleanly on ppc64le the first time.

All three CUDA *patches* live in the ONNX Runtime repository.

## What genai actually contributes on the GPU

It is easy to assume genai is a thin wrapper that forwards everything to ONNX
Runtime. It is not, and the wheel shows it:

| | CPU wheel | CUDA wheel |
|---|---|---|
| `libonnxruntime-genai.so` | 7.5 MB | 7.5 MB — byte-identical, links no CUDA |
| `libonnxruntime-genai-cuda.so` | — | **11.0 MB** |

The CUDA library links `libcudart`, `libcublas` and `libcublasLt`, and carries a
**9.7 MB `.nv_fatbin`** — compiled GPU device code. Its symbols do not show up
under `nm -D` because the library exports a narrow interface and keeps the
kernels internal; the fatbin section is the proof.

Source in `src/cuda/`, 21 files:

```
cuda_sampling.cu            sampling (top-p, temperature)
cuda_topk.cu + 9 .cuh       top-k, with nine different strategies
beam_search_scorer_cuda.cu  beam search scoring
beam_search_topk.cu         beam search top-k
search_cuda.cu              the generation loop on device
model_kernels.cu            assorted model-adjacent kernels
```

**None of it is the model forward pass** — that stays in ONNX Runtime. This is
the generation loop: choose the next token, maintain the beams, update the KV
cache bookkeeping.

### Why it has to be on the GPU

Latency, not throughput. If sampling ran on the host, every token would require
copying the logits back — 128,256 floats (~513 KB) for Llama 3.2, 32,064 for
Phi-3. The bandwidth is affordable at 178 tok/s; the **synchronization point on
every token** is not: it drains the GPU pipeline and serializes what should
overlap.

Nine top-k implementations in one directory is a reasonable proxy for how much
this matters — that kernel sits directly on the per-token critical path.

### What this means for the port

The division of labour is clean, and it is why this repository needed no CUDA
patch:

| | Where it runs | Who fixed it for POWER9 |
|---|---|---|
| Transformer layers, GEMMs | ONNX Runtime CUDA EP | the three ORT patches |
| Sampling, top-k, beam search | genai's own CUDA kernels | nothing needed |
| Tokenization, chat template | genai, host side | nothing needed |

## Building

```bash
# with the CUDA ORT already built and assembled into ort-home-cuda
bash build_genai_cuda_power9.sh
```

Produces `onnxruntime_genai_cuda-0.15.0-cp312-cp312-linux_ppc64le.whl`.

## Running

```bash
export CUDA_VISIBLE_DEVICES=0,1      # only GPUs 0 and 1 initialize on this box
python tests/test_genai_cuda.py
```

`gpu/download_cuda_model.py` fetches the CUDA variant of Phi-3-mini.

### Turn on CUDA graph

The `genai_config.json` shipped with CUDA models has it off:

```json
"provider_options": [{"cuda": {"enable_cuda_graph": "1"}}]
```

Worth **+8.8%** on decode, with byte-identical output.

### Do not batch between 2 and 7

The int4 CUDA kernel is strictly M=1. Batch 1 delivers more total throughput
than batch 2 or 4; only from batch 8 does batching pay off, and per-sequence
throughput drops 5× either way. Details in the ONNX Runtime repository.

## Verify you are actually on the GPU

A broken build lists `CUDAExecutionProvider` and runs on the CPU without saying
so. genai hides the session behind its own API, so the practical check is the
throughput itself: if decode is in the tens of tok/s rather than the hundreds,
you are on the CPU. Confirm with the ONNX Runtime-level test
(`test_ort_cuda.py`), which asserts the provider and counts node assignment.
