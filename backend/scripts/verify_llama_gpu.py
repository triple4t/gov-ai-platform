"""Print whether llama.cpp was built with GPU offload (run after CUDA reinstall)."""
from llama_cpp import llama_supports_gpu_offload

print("llama_supports_gpu_offload =", llama_supports_gpu_offload())
