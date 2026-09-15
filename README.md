# onnxruntime-genai 0.15.0 for POWER9 (ppc64le)

`onnxruntime-genai` ships no prebuilt wheels for ppc64le, so it has to be built
from source. This repository contains the four patches needed to make that
build succeed, the build script, and an end-to-end generation test.

The library itself is architecture-neutral C++ — tokenization, the generation
loop, KV cache management, sampling. It does no math of its own; it delegates to
ONNX Runtime. Every problem found here was in **build plumbing**, not in
execution code.

> **This is only half of the port.** genai compiles and produces correct output
> with these four patches, but generation runs at 0.13 tok/s unless ONNX Runtime
> itself has an int4 MLAS kernel for POWER. See
> [ONNXRuntime-1.26-Power9](../ONNXRuntime-1.26-Power9) — that repository
> contains the kernel, and is what takes this from 0.13 to 10.5 tok/s.

## The four patches

| # | Patch | ppc64le-specific? |
|---|---|---|
| 1 | `target_platform.cmake` recognises `ppc64le` | yes |
| 2 | onnxruntime-extensions: allow the system zlib on PowerPC | yes |
| 3 | `build.py` honours `--ort_home` for the C examples | **no — generic upstream bug** |
| 4 | Make the extensions patch step idempotent | no — side effect of patch 2 |

### 1. Architecture not recognised

```
CMake Error at cmake/target_platform.cmake:63 (message):
  Unsupported architecture.  CMAKE_SYSTEM_PROCESSOR: ppc64le
```

The file already had a PowerPC branch — its header says it normalizes to
"x64, arm64, or powerpc". But the test was `MATCHES "powerpc"`, which is what
AIX reports. Linux on POWER reports `ppc64le` (or `ppc64` on big-endian), so the
branch never matched on Linux. The support existed; only the AIX path had ever
been exercised.

### 2. zlib version pin

```
operators/vision/image_encoder.hpp:10:2: error: #error "stopped"
note: Invalid zlib version:  0x1320
```

`onnxruntime-extensions` vendors dlib, whose bundled libpng only carries ARM/NEON
optimizations. Its `ext_imgcodecs.cmake` therefore already branches on PowerPC
and fetches upstream libpng instead — which resolves zlib with `find_package`,
i.e. the **system** zlib.

`image_encoder.hpp` was never updated for that branch. It still hard-asserts
`ZLIB_VERNUM == 0x12b0` (dlib's vendored zlib 1.2.11), so any modern system zlib
fails the build. The patch skips the pin under `__powerpc__`, where using the
system zlib is the intended design.

Applied through FetchContent's `PATCH_COMMAND` so it survives clean rebuilds —
editing files under `_deps/` would be wiped.

> RHEL 8 ships exactly zlib 1.2.11. The upstream PowerPC support (PRs #1041 and
> #1051, April–May 2026) was almost certainly developed on a machine where the
> assert passed by coincidence. Worth mentioning if you upstream this — the
> maintainer will look at the assert and think "but it works here".

### 3. `--ort_home` ignored for the C examples

```
examples/c/src/common.h:18:10: fatal error:
        onnxruntime_cxx_api.h: No such file or directory
```

`build_examples()` hardcoded `REPO_ROOT/ort/{include,lib}` — the location used
when cmake downloads ONNX Runtime itself — ignoring the `--ort_home` argument
that the rest of the build respects.

**This is not a ppc64le bug.** It breaks on any platform that builds against its
own ONNX Runtime, x86 included. It only goes unnoticed because platforms with a
prebuilt package rarely need `--ort_home`. On ppc64le there is no alternative.

This is the most readily upstreamable finding here.

### 4. Idempotent patching

CMake re-runs `PATCH_COMMAND` on every reconfigure, so the second configure hit
`patch does not apply` on an already-patched tree. Resetting the checkout before
applying makes the step repeatable. Not an upstream bug — a side effect of
patch 2, recorded because it is a classic `PATCH_COMMAND` trap.

## Building

Requires a POWER9 build of ONNX Runtime 1.26.0 first — see the sibling
repository.

```bash
conda create -n onnx_build -c conda-forge python=3.12 "cmake=3.31.*" ninja \
        numpy zlib gcc_linux-ppc64le=13.3 gxx_linux-ppc64le=13.3
pip install "wheel==0.47.0" "pybind11==3.0.4" requests

git clone --recursive -b v0.15.0 https://github.com/microsoft/onnxruntime-genai.git
cd onnxruntime-genai && git am ../patches/*.patch

bash build/build_genai_cpu_power9.sh
```

`zlib` in the environment is not optional — it is what the PowerPC libpng branch
resolves against.

The script passes `--no_telemetry`. That is a choice, not a fix: telemetry is on
by default and pulls in Microsoft's 1DS client along with curl and mbedtls.
Disabling it removes a dependency tree that would otherwise need validating on
ppc64le.

## Validation

- **89/89** in the project's own C++ suite, 0 failures. The 22 skipped tests need
  TensorRT or ASR model data. Several are real end-to-end generation tests
  (`EndToEndPhi`, `GreedySearchGptFp32`, `BeamSearchGptFp32`) that compare
  against token sequences produced on x86 — effectively cross-platform golden
  values.
- Real generation with Phi-3-mini and Llama 3.2 1B: coherent output,
  deterministic under greedy decoding, correct arithmetic.

## Runtime notes

Set `intra_op_num_threads` between 40 and 80 in the model's `genai_config.json`,
under `session_options`. The machine has 160 logical CPUs and ONNX Runtime uses
all of them by default, which is 2.4× slower than the optimum on the small
per-token GEMMs of decoding.

The wheel requires a conda-forge `libstdcxx-ng` at runtime.

## Repository layout

```
patches/   the four commits, as git-format-patch files
build/     build script, with toolchain rationale in the header
tests/     end-to-end generation and determinism check
docs/      build troubleshooting write-up
```

---

UFCG / IBM — `#ibm-multiarq`
Built and validated on an IBM Power System AC922 (8335-GTH), 2× POWER9,
RHEL 8, ppc64le.
