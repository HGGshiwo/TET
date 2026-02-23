# https://huggingface.co/learn/cookbook/en/fine_tuning_vlm_grpo_trl
import json
import torch
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Qwen2TokenizerFast,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLProcessor,
)
from data_utils import (
    Prompt,
    format_output,
    generate_dataset,
    parse_multi_choice_response,
    prepare_inputs,
)
from utils import load_data
from peft import LoraConfig, get_peft_model
from trainer.grpo_trainer import GRPOConfig, Qwen2VLGRPOTrainer
from typing import List, Optional, Dict, Any

data_cfg_path = r"D:\work\实时对话\TET\train\config\dataset_cfg.yml"
model_id = r"D:\work\实时对话\TET\train\outputs\sft8_2-merge"

OUTPUT_PATH = r"D:\work\实时对话\TET\train\outputs\sft8_2-merge-r1"
PROMPT_TYPE = "v1_5"
EPOCH_NUM = 3

prompt = Prompt.create(PROMPT_TYPE)
data_cfg = load_data(data_cfg_path)

train_dataset, eval_dataset = generate_dataset(
    dataset_cfg=data_cfg,
    prompt=prompt,
    # filter=lambda data: parse_multi_choice_response(data["answer"]) == data["truth"],
    split_test=False,
)


def format(data: Dict[str, Any]):
    """适配trainer的数据结构"""
    data["message"] = [msg for msg in data["message"] if msg["role"] != "assistant"]
    data["prompt"] = data["message"]
    data["data_type"] = "video"
    return data


train_dataset = train_dataset.map(format, desc="[train]format data structure")
eval_dataset = eval_dataset.map(format, desc="[eval]format data structure")


# Model and processor configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
    use_cache=True,
)
processor = Qwen2_5_VLProcessor.from_pretrained(model_id)
# Set padding side to left for decoder-only architecture
processor.tokenizer.padding_side = "left"

# Configure LoRA for model adaptation
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=8,
    bias="none",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    task_type="CAUSAL_LM",
)

# Configure training arguments using GRPOConfig
training_args = GRPOConfig(
    output_dir=OUTPUT_PATH,
    num_train_epochs=EPOCH_NUM,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=1,
    gradient_checkpointing=False,
    optim="adamw_torch_fused",
    learning_rate=1e-5,
    lr_scheduler_type="constant",
    remove_unused_columns=False,  # to access the solution column in accuracy_reward
    logging_steps=10,
    eval_steps=100,
    eval_strategy="steps",  # Evaluation strategy
    save_strategy="steps",  # Strategy for saving the model
    save_steps=100,
    metric_for_best_model="reward_mean",  # Metric to evaluate the best model
    greater_is_better=True,  # Lower metric values are better
    load_best_model_at_end=True,
    bf16=True,
    tf32=True,  # Use TensorFloat-32 precision
    max_grad_norm=0.3,  # Maximum gradient norm for clipping
    warmup_ratio=0.03,  # Warmup ratio for learning rate
    report_to="tensorboard",
    push_to_hub=False,  # Do not push the model to Hugging Face Hub
    gradient_checkpointing_kwargs={
        "use_reentrant": False
    },  # Gradient checkpointing options
    use_vllm=False,
    num_generations=8,
    generation_batch_size=4,  # not use
    temporal=False,
    len_control=False,
)


def reward_func(completions: list, truth: list[str], **kwargs):
    """Reward function that checks if the completion has a specific format."""
    results = []
    for completion, sol in zip(completions, truth):
        content = completion[0]["content"]
        reward = 0
        try:
            res = prompt.format_output(content)
            if not all([key in res for key in ["reasoning", "keyframes", "answer"]]):
                reward = -1
            elif parse_multi_choice_response(res["answer"]) == sol:
                reward = 1
        except Exception:
            pass
        results.append(reward)
    return results


trainer = Qwen2VLGRPOTrainer(
    reward_funcs=[reward_func],
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=processor,
    peft_config=peft_config,
)

trainer.train()
trainer.save_model(training_args.output_dir)
