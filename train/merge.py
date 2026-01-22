import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
from peft import LoraConfig, PeftModel, PeftConfig

model_id = r"D:\work\实时对话\TET\train\outputs\sft7\checkpoint-4800"
OUTPUT_PATH = r"D:\work\实时对话\TET\train\outputs\sft7-4800-merge"

config = PeftConfig.from_pretrained(model_id)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.base_model_name_or_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
)
model = PeftModel.from_pretrained(model, model_id)
processor = AutoProcessor.from_pretrained(model_id)
# Set padding side to left for decoder-only architecture
processor.tokenizer.padding_side = "left"

merged_model = model.merge_and_unload(progressbar=True)

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