# Build and validation results

Measured on an IBM Power System AC922 (8335-GTH): 2× POWER9, 20 cores per
socket, SMT4 (160 logical CPUs), 512 GB RAM, RHEL 8, ppc64le.

Toolchain: conda-forge GCC 13.3, CMake 3.31, Python 3.12.

> **Scope.** This document covers what the genai port itself delivers: a
> working build, a correct library, and real text generation. It deliberately
> does **not** carry the throughput numbers — those come from the int4 MLAS
> kernel in [ONNXRuntime-1.26-Power9](https://github.com/llm-pt-ibm/ONNXRuntime-1.26-Power9),
> not from anything in this repository. genai delegates all math to ONNX
> Runtime.

## Artifacts

| Wheel | Size |
|---|---|
| `onnxruntime_genai-0.15.0-cp312-cp312-linux_ppc64le.whl` | 3.0 MB |
| `onnxruntime-1.26.0-cp312-cp312-linux_ppc64le.whl` (dependency) | 21.9 MB |

Also built: `libonnxruntime-genai.so` (7.5 MB), the Python extension module
`onnxruntime_genai.cpython-312-powerpc64le-linux-gnu.so`, and the C example
programs — the last of which only link after patch 3.

Both wheels require a conda-forge `libstdcxx-ng` at runtime. RHEL 8's system
libstdc++ is from the GCC 8 era and too old for a GCC 13 build.

Telemetry is disabled (`--no_telemetry`), so the binaries do not carry
Microsoft's 1DS client, curl or mbedtls.

## C++ unit test suite

The project's own suite, run against the ppc64le build:

```
[==========] 111 tests from 12 test suites ran. (6447 ms total)
[  PASSED  ] 89 tests.
[  SKIPPED ] 22 tests
```

**Zero failures.** The 12 suites are `CAPITests`, `ModelTests`,
`ValidationTests`, `SamplingTests`, `SamplingBenchmarks`, `NarrowTest`,
`TelemetryContextTests`, `TelemetryEnvironmentTests`, `TelemetryRedactionTest`,
`ValidateConfigPathTest`, `VisionStateTypeHierarchy` and `WorkerThreadTest`.

### What the 22 skips are

All are skipped for missing hardware or missing model data — none are skipped
because of ppc64le:

| Count | Group | Why skipped |
|---|---|---|
| 10 | `*NvTensorRtRtx` | needs NVIDIA TensorRT-RTX; this is a CPU build |
| 9 | `StreamingASR*` / `*Vad*` | needs speech model data not fetched |
| 3 | `ParakeetTdt*` | needs the Parakeet ASR model |

The suite also reports 6 tests disabled upstream, unrelated to this port.

### The tests that matter most

Several of the passing tests are full generation runs that compare against
token sequences produced on x86 — effectively cross-platform golden values,
which is exactly what catches endianness and signedness bugs:

```
[  OK ] CAPITests.EndToEndPhi
[  OK ] CAPITests.EndToEndPhiBatch
[  OK ] CAPITests.EndToEndPhiEOSPAD
[  OK ] ModelTests.GreedySearchGptFp32
[  OK ] ModelTests.BeamSearchGptFp32
[  OK ] CAPITests.GreedySearchGptFp32CAPI
[  OK ] CAPITests.BatchedRewindGptFp32CAPI
[  OK ] CAPITests.GreedySearchLfm2Fp32CAPI
[  OK ] CAPITests.TokenizerCAPI
[  OK ] CAPITests.ChatTemplate
[  OK ] CAPITests.SetLogitsCAPI / GetLogitsCAPI
```

Tokenization, the generation loop, KV cache rewind, batching, beam search,
sampling and chat templates all behave identically to the reference platform.

## End-to-end generation

Two models were run through `tests/test_genai_e2e.py`:

| Model | Format | Source |
|---|---|---|
| Phi-3-mini-4k-instruct | int4 RTN block-32 | `microsoft/Phi-3-mini-4k-instruct-onnx` |
| Llama 3.2 1B Instruct | int4 RTN block-32, acc-level-4 | `onnx-community/Llama-3.2-1B-Instruct-GENAI-ONNX` |

Checks performed:

- **Coherence** — the model produces valid, on-topic text. Asked for a Python
  function it returns working code with a docstring; asked for arithmetic it
  answers `17 * 23` → `391`.
- **Determinism** — greedy decoding reproduces byte-identical output across
  runs. Divergence would indicate uninitialised memory or a race in the kernel
  below.
- **Non-empty output** — guards against a silently broken decode loop.

All passed.

> One observation worth passing on: text generation quality depends on applying
> the model's **chat template** (`<|user|> … <|assistant|>`). Feeding raw text
> to an instruct model produces rambling, off-topic output that looks like a
> broken port but is only a formatting mistake. genai handles this natively —
> `CAPITests.ChatTemplate` covers it.

## Runtime configuration

ONNX Runtime uses every logical CPU by default. On this machine that is 160,
and it is consistently the worst choice for token-by-token decoding, where the
per-token GEMMs are too small to amortize the synchronization cost.

Measured on Llama 3.2 1B (int4), varying only the thread count:

| `intra_op_num_threads` | tok/s |
|---|---|
| 20 | 21.61 |
| 40 | 21.97 |
| **80** | **23.49** |
| 160 (default) | 9.65 |

**Set it explicitly**, in the model's `genai_config.json`:

```json
{
  "model": {
    "decoder": {
      "session_options": {
        "intra_op_num_threads": 80
      }
    }
  }
}
```

Anything between 40 and 80 is reasonable on this machine. The optimum shifts
with model size and with how efficient the underlying kernel is — sweep it
before publishing any benchmark rather than trusting a number from another
model.

## What this validation does and does not cover

**Covered:** the library builds, links, loads, tokenizes, manages KV cache,
samples, applies chat templates, and generates correct and deterministic text
for the model architectures exercised by the suite and the two models above.

**Not covered:**

- **Multimodal paths.** Vision and audio operators compile (that is what the
  zlib patch is about), but no multimodal model was run. The ASR tests are
  skipped for missing data.
- **Anything GPU.** This is a CPU build; the TensorRT tests are skipped.
- **Model architectures beyond those tested.** genai advertises Llama, Mistral,
  Phi, Gemma and Granite; only Phi-3 and Llama 3.2 were exercised here.
- **Throughput.** As noted at the top — the numbers that make generation usable
  come from the int4 kernel in the ONNX Runtime repository. Without it, this
  same build runs at 0.13 tok/s: correct, and unusably slow.
