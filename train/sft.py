# https://connectaman.hashnode.dev/fine-tuning-the-qwen25-7b-vl-instruct-model-a-comprehensive-guide
import numpy as np
import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer

from data_utils import (
    Prompt,
    PromptV1_5,
    generate_dataset,
    parse_multi_choice_response,
    prepare_inputs,
)
from utils import load_data
from copy import deepcopy

data_cfg_path = r"D:\work\实时对话\TET\train\config\dataset_cfg.yml"
model_id = r"D:\models\Qwen2.5-VL-7B-Instruct"
OUTPUT_PATH = r"D:\work\实时对话\TET\train\outputs\sft3"
EPOCH_NUM = 3
PROMPT_TYPE = "v1_5"

assistant_start_token = "<assistant_start>"


def collate_fn(examples):
    """
    全封装版Data collator：
    1. 函数内临时添加自定义token，返回前恢复tokenizer原始状态，不污染全局
    2. assistant消息开头插入自定义token，encode后按该token切分label（前方标-100）
    3. 删除插入的自定义token，input/attention_mask/labels同步删除保证维度一致
    4. 无外部初始化操作，直接调用，不影响其他地方的编码逻辑
    """
    # -------------------------- 步骤1：保存tokenizer原始状态，临时添加自定义token --------------------------
    _processor = deepcopy(processor)
    tokenizer = _processor.tokenizer
    # 保存原始的额外特殊token列表（用于后续恢复，核心！）
    original_add_special_tokens = tokenizer.additional_special_tokens.copy()
    try:
        # 临时为tokenizer添加自定义token（仅在当前函数内生效，后续会恢复）
        if assistant_start_token not in original_add_special_tokens:
            new_special_tokens = {"additional_special_tokens": [assistant_start_token]}
            tokenizer.add_special_tokens(new_special_tokens)
        # 获取自定义token的id（临时添加后才会有正确的id，非unk）
        assistant_start_token_id = tokenizer.convert_tokens_to_ids(
            assistant_start_token
        )
        # 断言token添加成功，避免id为unk（排查错误用）
        assert (
            assistant_start_token_id != tokenizer.unk_token_id
        ), f"自定义token {assistant_start_token} 添加失败，id为unk"

        # -------------------------- 步骤2：为assistant消息插入自定义token --------------------------
        for example in examples:
            for msg in example["message"]:
                if msg["role"] == "assistant":
                    msg["content"][0]["text"] = (
                        assistant_start_token + msg["content"][0]["text"]
                    )

        # -------------------------- 步骤3：执行原始编码逻辑 --------------------------
        batch = prepare_inputs(_processor, examples, add_generation_prompt=False)

        # -------------------------- 步骤4：初始化labels，标记pad_token为-100 --------------------------
        labels = batch["input_ids"].clone()
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100

        # -------------------------- 步骤5：基于自定义token切分label，前方（含自身）标为-100 --------------------------
        # 找到每个样本中自定义token的位置
        assistant_mask = labels == assistant_start_token_id
        assert assistant_mask.any(
            dim=1
        ).all(), "部分样本未找到assistant_start_token，请检查数据中是否有assistant消息"
        target_positions = assistant_mask.int().argmax(
            dim=1
        )  # 每个样本的token首次出现位置

        # 生成位置索引，标记需要mask的区域（<= token位置的全部标-100）
        batch_size, seq_len = labels.shape[:2]
        positions = (
            torch.arange(seq_len, device=labels.device)
            .unsqueeze(0)
            .expand(batch_size, seq_len)
        )
        mask_positions = positions <= target_positions.unsqueeze(1)
        labels[mask_positions] = -100

        # -------------------------- 步骤6：删除插入的自定义token，三张量同步操作 --------------------------
        # 核心：input_ids/attention_mask/labels必须同步删除该位置，保证维度完全一致
        new_input_ids, new_attention_mask, new_labels = [], [], []
        for idx in range(batch_size):
            token_pos = target_positions[idx].item()  # 当前样本的自定义token位置
            # 删去token_pos位置的元素，拼接前后部分
            input_ids_i = torch.cat(
                [
                    batch["input_ids"][idx, :token_pos],
                    batch["input_ids"][idx, token_pos + 1 :],
                ]
            )
            attention_mask_i = torch.cat(
                [
                    batch["attention_mask"][idx, :token_pos],
                    batch["attention_mask"][idx, token_pos + 1 :],
                ]
            )
            labels_i = torch.cat(
                [labels[idx, :token_pos], labels[idx, token_pos + 1 :]]
            )
            # 加入新列表
            new_input_ids.append(input_ids_i)
            new_attention_mask.append(attention_mask_i)
            new_labels.append(labels_i)

        # 转回batch级张量，替换原batch中的值
        batch["input_ids"] = torch.stack(new_input_ids)
        batch["attention_mask"] = torch.stack(new_attention_mask)
        batch["labels"] = torch.stack(new_labels)

    # -------------------------- 关键：无论是否报错，都恢复tokenizer原始状态 --------------------------
    finally:
        # 强制将tokenizer的额外特殊token恢复为原始状态，彻底避免污染全局
        pass

    # 返回处理后的batch
    return batch


prompt = Prompt.create(PROMPT_TYPE)
data_cfg = load_data(data_cfg_path)
train_dataset, eval_dataset = generate_dataset(
    prompt=prompt,
    dataset_cfg=data_cfg,
    # filter=lambda data: parse_multi_choice_response(data["answer"]) == data["truth"],
    split_test=False,
)


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
)
processor = AutoProcessor.from_pretrained(model_id)
# Set padding side to left for decoder-only architecture
processor.tokenizer.padding_side = "left"
special_tokens = prompt.get_special_tokens()
trainable_token_indices = []
if len(special_tokens) != 0:
    print(f"add additional tokens: {special_tokens}")
    processor.tokenizer.add_tokens(special_tokens)
    model.resize_token_embeddings(len(processor.tokenizer))
    trainable_token_indices = tokenizer.convert_tokens_to_ids(special_tokens)

# Configure LoRA for model adaptation
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=16,
    bias="none",
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
    modules_to_save=["lm_head"],
    trainable_token_indices=trainable_token_indices,
)

# Apply PEFT model adaptation and print trainable parameters
peft_model = get_peft_model(model, peft_config)
peft_model.print_trainable_parameters()

# Configure training arguments using SFTConfig
training_args = SFTConfig(
    output_dir=OUTPUT_PATH,  # Directory to save the model
    num_train_epochs=EPOCH_NUM,  # Number of training epochs
    per_device_train_batch_size=1,  # Training batch size per device
    per_device_eval_batch_size=4,  # Evaluation batch size per device
    gradient_accumulation_steps=4,  # Number of steps to accumulate gradients
    gradient_checkpointing=True,  # Enable gradient checkpointing for memory efficiency
    optim="adamw_torch_fused",  # Optimizer type
    learning_rate=2e-4,  # Learning rate for training
    lr_scheduler_type="constant",  # Learning rate scheduler type
    logging_steps=10,  # Interval (in steps) for logging
    eval_steps=100,  # Interval (in steps) for evaluation
    eval_strategy="steps",  # Evaluation strategy
    save_strategy="steps",  # Strategy for saving the model
    save_steps=100,  # Interval (in steps) for saving
    metric_for_best_model="eval_loss",  # Metric to evaluate the best model
    greater_is_better=False,  # Lower metric values are better
    load_best_model_at_end=True,  # Load the best model after training
    bf16=True,  # Use bfloat16 precision
    tf32=True,  # Use TensorFloat-32 precision
    max_grad_norm=0.3,  # Maximum gradient norm for clipping
    warmup_ratio=0.03,  # Warmup ratio for learning rate scheduler
    report_to="tensorboard",  # Reporting via Weights & Biases
    push_to_hub=False,  # Do not push the model to Hugging Face Hub
    gradient_checkpointing_kwargs={
        "use_reentrant": False
    },  # Gradient checkpointing options
    dataset_text_field="",  # Text field in the dataset (if applicable)
    dataset_kwargs={"skip_prepare_dataset": True},  # Additional dataset options
    # max_seq_length=1024  # Uncomment to set maximum sequence length for input
)

# Do not remove unused columns from the dataset
training_args.remove_unused_columns = False

# Create the trainer for fine-tuning the model
trainer = SFTTrainer(
    model=peft_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collate_fn,
    processing_class=processor,
    # tokenizer=processor.tokenizer,
)

# Start training
trainer.train()

# Save the model checkpoint (with sharding as needed)
trainer.model.save_pretrained(training_args.output_dir, max_shard_size="4GB")
# Save the tokenizer and processor configurations
processor.tokenizer.save_pretrained(training_args.output_dir)
processor.save_pretrained(training_args.output_dir)
