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

from data_utils import generate_dataset, parse_multi_choice_response, prepare_inputs
from utils import load_data


data_cfg_path = r"D:\work\实时对话\TET\train\config\dataset_cfg.yml"
model_id = r"D:\models\Qwen2.5-VL-7B-Instruct"
OUTPUT_PATH = r"D:\work\实时对话\TET\train\outputs\sft7"
EPOCH_NUM = 3
PROMPT_TYPE = "v1"

def min_nonzero_pos(x):
    """x: mask, shape of (batch_size, sequence)
    return non zero position in x, shape of (batch_size)
    if more than one posiiton, return the smallest ones
    """
    assert (x.sum(dim=1) > 0).all(), "No valid posiiton"
    # 用一个很大的数填充0的位置
    x_masked = x.clone()
    x_masked[~x] = float("inf")
    min_values, min_indices = x_masked.min(dim=1)
    return min_indices


def collate_fn(examples):
    """
    Data collator to prepare a batch of examples.

    This function applies the chat template to texts, processes the images,
    tokenizes the inputs, and creates labels with proper masking.
    """
    # Apply chat template to each example (no tokenization here)
    batch = prepare_inputs(processor, examples, add_generation_prompt=False)

    # Create labels by cloning input_ids and mask the pad tokens
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    assistant_token = processor.tokenizer.convert_tokens_to_ids("assistant")
    assistant_mask = labels == assistant_token

    assert assistant_mask.any(dim=1).all(), "No assistant token found"
    target_positions = assistant_mask.int().argmax(dim=1)
    batch_size, seq_len = labels.shape[:2]
    positions = torch.arange(seq_len, device=labels.device).unsqueeze(0)
    positions = positions.expand(batch_size, seq_len)
    mask_positions = positions <= target_positions.unsqueeze(1)
    labels[mask_positions] = -100

    batch["labels"] = labels
    return batch


data_cfg = load_data(data_cfg_path)
train_dataset, eval_dataset = generate_dataset(
    dataset_cfg=data_cfg,
    prompt_type=PROMPT_TYPE,
    # filter=lambda data: parse_multi_choice_response(data["answer"]) == data["truth"],
    split_test=False,
)

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
