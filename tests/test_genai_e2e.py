#!/usr/bin/env python
"""
End-to-end LLM generation with onnxruntime-genai on ppc64le / POWER9.

Runs Phi-3-mini-4k-instruct (int4 RTN, block 32) and checks that generation
is coherent and deterministic. Greedy decoding is deterministic by
construction, so the same prompt must give the same tokens every time - a
sanity check that no uninitialised memory or racy kernel is in play.

Usage: python test_genai_e2e.py [model_dir]
"""

import sys
import time

import onnxruntime_genai as og

MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else \
    "/root/onnx/models/phi3-mini/cpu_and_mobile/cpu-int4-rtn-block-32"

PROMPTS = [
    "Explain what a POWER9 processor is, in two sentences.",
    "Write a Python function that reverses a string.",
    "What is 17 * 23? Answer with just the number.",
]


def build_prompt(text):
    return f"<|user|>\n{text}<|end|>\n<|assistant|>\n"


def generate(model, tokenizer, prompt, max_new=120):
    params = og.GeneratorParams(model)
    params.set_search_options(max_length=max_new + 256, do_sample=False)

    generator = og.Generator(model, params)
    generator.append_tokens(tokenizer.encode(build_prompt(prompt)))

    out = []
    stream = tokenizer.create_stream()
    t0 = time.time()
    n = 0
    while not generator.is_done() and n < max_new:
        generator.generate_next_token()
        out.append(stream.decode(generator.get_next_tokens()[0]))
        n += 1
    dt = time.time() - t0
    return "".join(out), n, n / dt if dt else 0.0


def main():
    print(f"onnxruntime-genai {og.__version__}")
    print(f"modelo: {MODEL_DIR}")
    print("=" * 70)

    t0 = time.time()
    config = og.Config(MODEL_DIR)
    model = og.Model(config)
    tokenizer = og.Tokenizer(model)
    print(f"modelo carregado em {time.time() - t0:.1f}s")
    print("=" * 70)

    failures = []
    first_run = {}

    for prompt in PROMPTS:
        text, n, tps = generate(model, tokenizer, prompt)
        first_run[prompt] = text
        print(f"\n>>> {prompt}")
        print(text.strip()[:400])
        print(f"    [{n} tokens, {tps:.1f} tok/s]")
        if not text.strip():
            failures.append(f"saída vazia para: {prompt}")

    # Determinism: greedy decoding must reproduce exactly.
    print("\n" + "=" * 70)
    print("checando determinismo (greedy deve repetir exatamente)...")
    for prompt in PROMPTS[:2]:
        text, _, _ = generate(model, tokenizer, prompt)
        if text != first_run[prompt]:
            failures.append(f"NÃO determinístico para: {prompt}")
            print(f"  DIVERGIU: {prompt}")
        else:
            print(f"  ok  {prompt[:50]}")

    print("=" * 70)
    if failures:
        for f in failures:
            print(f"FALHOU  {f}")
        return 1
    print("geração ponta a ponta OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
