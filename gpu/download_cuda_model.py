from huggingface_hub import snapshot_download
p = snapshot_download(
    repo_id="microsoft/Phi-3-mini-4k-instruct-onnx",
    allow_patterns=["cuda/cuda-int4-rtn-block-32/*"],
    local_dir="/root/onnx/models/phi3-mini",
)
print("OK ->", p)
