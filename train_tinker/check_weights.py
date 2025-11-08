# pip install safetensors torch
from safetensors.torch import load_file

tensors = load_file("train_tinker/adapters/my_adapter/adapter_model.safetensors", device="cpu")
print(list(tensors.keys())[:10])       
w = tensors["base_model.model.model.layers.0.self_attn.k_proj.lora_B.weight"] 
print(w.shape, w.dtype)
print(w[:2, :5])  