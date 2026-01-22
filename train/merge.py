import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model

model_id = r"D:\work\实时对话\TET\train\outputs\sft7\checkpoint-4800"
OUTPUT_PATH = r"D:\work\实时对话\TET\train\outputs\sft7-4800-merge"


model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
processor = AutoProcessor.from_pretrained(model_id)
# Set padding side to left for decoder-only architecture
processor.tokenizer.padding_side = "left"

# Configure LoRA for model adaptation
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=16,
    bias="none",
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)

# Apply PEFT model adaptation and print trainable parameters
peft_model = get_peft_model(model, peft_config)

merged_model = peft_model.merge_and_unload(progressbar=True)

# https://huggingface.co/docs/peft/main/en/developer_guides/checkpoint#convert-to-a-transformers-model
merged_model._hf_peft_config_loaded = False
merged_model.save_pretrained(
    OUTPUT_PATH,
    safe_serialization=True  # 安全序列化，兼容更多环境
)
# Save the tokenizer and processor configurations
processor.tokenizer.save_pretrained(OUTPUT_PATH)
processor.save_pretrained(OUTPUT_PATH)
print(f"✅ 合并完成！完整模型已保存至：{OUTPUT_PATH}")