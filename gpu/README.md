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

**The same four CPU patches apply** — they are build plumbing (architecture
detection, the zlib pin, `--ort_home`, patch idempotency) and are orthogonal to
the execution provider. There is no genai-specific CUDA patch.

All the CUDA-side work lives in the ONNX Runtime repository.

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
