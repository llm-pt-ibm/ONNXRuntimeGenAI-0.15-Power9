"""Teste end-to-end do onnxruntime-genai com CUDA na Tesla V100 (Power9/ppc64le).

Mede prefill e decode separadamente: sao regimes diferentes (o prefill e limitado por
compute, o decode por banda de memoria), e reportar so a media esconde isso.

O chat template e obrigatorio -- passar texto cru para um modelo instruct gera saida
incoerente e parece "modelo quebrado em POWER" quando e so formatacao de prompt.
"""
import sys
import time

import onnxruntime_genai as og

MODEL = "/root/onnx/models/phi3-mini/cuda/cuda-int4-rtn-block-32"
PROMPT = "Explique em um paragrafo por que a arquitetura POWER9 usa little-endian."
MAX_NEW = 100

print("onnxruntime_genai:", og.__version__)

config = og.Config(MODEL)
t0 = time.time()
model = og.Model(config)
load_s = time.time() - t0
print(f"modelo carregado em {load_s:.1f}s")

tokenizer = og.Tokenizer(model)

# Chat template do Phi-3.
prompt = f"<|user|>\n{PROMPT}<|end|>\n<|assistant|>\n"
input_tokens = tokenizer.encode(prompt)
n_prompt = len(input_tokens)

params = og.GeneratorParams(model)
params.set_search_options(max_length=n_prompt + MAX_NEW, do_sample=False)
generator = og.Generator(model, params)

# --- prefill ---
t0 = time.time()
generator.append_tokens(input_tokens)
prefill_s = time.time() - t0

# --- decode ---
stream = tokenizer.create_stream()
out = []
t0 = time.time()
n_new = 0
while not generator.is_done() and n_new < MAX_NEW:
    generator.generate_next_token()
    tok = generator.get_next_tokens()[0]
    out.append(stream.decode(tok))
    n_new += 1
decode_s = time.time() - t0

text = "".join(out)
print("\n--- saida ---")
print(text)
print("--- fim ---\n")

print(f"prompt: {n_prompt} tokens | gerados: {n_new} tokens")
print(f"prefill: {prefill_s * 1000:.0f} ms  ({n_prompt / prefill_s:.1f} tok/s)")
print(f"decode:  {decode_s:.2f} s   ({n_new / decode_s:.1f} tok/s)")

# Guardas: uma saida vazia ou degenerada nao pode passar por sucesso.
assert n_new >= MAX_NEW // 2, f"gerou poucos tokens: {n_new}"
assert len(text.strip()) > 40, f"saida vazia ou curta demais: {text!r}"
if len(set(text.split())) < 5:
    sys.exit(f"saida degenerada (repeticao): {text!r}")
print("\nOK: geracao end-to-end na GPU.")
