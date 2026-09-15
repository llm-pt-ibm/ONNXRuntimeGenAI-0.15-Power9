# Provenance and licensing

The patches in `patches/` are derived from **onnxruntime-genai**, licensed under
the MIT License (Copyright (c) Microsoft Corporation).

Patch 2 additionally carries a change to **onnxruntime-extensions** (also MIT,
Microsoft). That change belongs upstream in the onnxruntime-extensions
repository rather than in genai — it is applied here through genai's
FetchContent `PATCH_COMMAND` only because that is where the build breaks.

Build scripts, tests and documentation are from the UFCG / IBM
multi-architecture team, released under the same terms.

Nothing here vendors or redistributes either project — the build scripts fetch
them from the official repositories at tags `v0.15.0` and the pinned extensions
commit.
