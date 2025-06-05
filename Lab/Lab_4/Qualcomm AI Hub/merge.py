from transformers import AutoModelForCausalLM
from peft import PeftModel
import torch

# 載入基礎模型
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct", torch_dtype=torch.bfloat16)

# 載入 LoRA 權重
lora_model = PeftModel.from_pretrained(base_model, "x21530317x/llama3.2-3B-instruct_lora_bf16")

# 合併權重
merged_model = lora_model.merge_and_unload()

# 儲存合併後的模型
merged_model.save_pretrained("llama3.2-3B-instruct-merged")
