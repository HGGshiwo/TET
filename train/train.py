# https://connectaman.hashnode.dev/fine-tuning-the-qwen25-7b-vl-instruct-model-a-comprehensive-guide
import torch
from datasets import load_dataset
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer

from train_utils import process_vision_info
from generate_dataset import generate_dataset


dataset_name = "egoschema_subset"

ANSWER_PATH = r"D:\work\实时对话\TET\outputs\qwenvl_test3\answer.jsonl"
model_id = r"D:\models\Qwen2.5-VL-7B-Instruct"
OUTPUT_PATH = r"D:\work\实时对话\TET\train\outputs\egoschema-sub-sft2"


def generate_text_from_sample(
    model, processor, sample, max_new_tokens=1024, device="cuda"
):
    """
    Generate output text from a single sample using the model and processor.

    Parameters:
        model: The vision-language generation model.
        processor: The processor to apply chat templates and tokenize inputs.
        sample: The input sample containing text and image data.
        max_new_tokens: Maximum number of new tokens to generate.
        device: Device to perform inference on.

    Returns:
        A string containing the generated output text.
    """
    # Apply chat template to sample (skip the system message)
    text_input = processor.apply_chat_template(
        sample[1:2], tokenize=False, add_generation_prompt=True
    )

    # Process visual inputs from the sample
    image_inputs, _ = process_vision_info(sample)

    # Prepare model inputs with text and image data, and move to the specified device
    model_inputs = processor(
        text=[text_input],
        images=image_inputs,
        return_tensors="pt",
    ).to(device)

    # Generate tokens with the model
    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)

    # Remove input tokens from generated output tokens
    trimmed_generated_ids = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # Decode the generated tokens into text
    output_text = processor.batch_decode(
        trimmed_generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0]


def collate_fn(examples):
    """
    Data collator to prepare a batch of examples.

    This function applies the chat template to texts, processes the images,
    tokenizes the inputs, and creates labels with proper masking.
    """
    # Apply chat template to each example (no tokenization here)
    examples = [example["message"] for example in examples]
    texts = [
        processor.apply_chat_template(example, tokenize=False) for example in examples
    ]
    # Process visual inputs for each example
    video_input = [process_vision_info(example)[1] for example in examples]

    # Tokenize texts and images into tensors with padding
    batch = processor(
        text=texts,
        videos=video_input,
        return_tensors="pt",
        padding=True,
    )

    # Create labels by cloning input_ids and mask the pad tokens
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    # Determine image token IDs to mask in the labels (model specific)
    # if isinstance(processor, Qwen2VLProcessor):
    #     image_tokens = [151652, 151653, 151655]
    # else:
    #     image_tokens = [
    #         processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    #     ]

    # # Mask image token IDs in the labels
    # for image_token_id in image_tokens:
    #     labels[labels == image_token_id] = -100

    assistant_token = processor.tokenizer.convert_tokens_to_ids("assistant")
    assistant_mask = labels == assistant_token
    batch_size, seq_len = labels.shape[:2]
    positions = torch.arange(seq_len, device=labels.device).unsqueeze(0)
    positions = positions.expand(batch_size, seq_len)
    
    target_positions = (assistant_mask * (positions + 1)).argmax(dim=1)
    
    mask_positions = positions < target_positions.unsqueeze(1)
    labels[mask_positions] = -100

    batch["labels"] = labels
    return batch


train_dataset, test_dataset, eval_dataset = generate_dataset(dataset_name, ANSWER_PATH)

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
processor.tokenizer.padding_side = 'left'

# Configure LoRA for model adaptation
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=8,
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
    num_train_epochs=4,  # Number of training epochs
    per_device_train_batch_size=1,  # Training batch size per device
    per_device_eval_batch_size=1,  # Evaluation batch size per device
    gradient_accumulation_steps=4,  # Number of steps to accumulate gradients
    gradient_checkpointing=True,  # Enable gradient checkpointing for memory efficiency
    optim="adamw_torch_fused",  # Optimizer type
    learning_rate=2e-4,  # Learning rate for training
    lr_scheduler_type="constant",  # Learning rate scheduler type
    logging_steps=10,  # Interval (in steps) for logging
    eval_steps=10,  # Interval (in steps) for evaluation
    eval_strategy="steps",  # Evaluation strategy
    save_strategy="steps",  # Strategy for saving the model
    save_steps=20,  # Interval (in steps) for saving
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
