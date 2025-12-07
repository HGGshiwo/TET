import torch
import os
import json
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from datasets import load_dataset
from generate_dataset import generate_dataset, format_output
from train_utils import process_vision_info
from transformers import BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import numpy as np
from tqdm import tqdm


dataset_name = "egoschema_subset"
# Model and processor setup
model_id = r"D:\work\实时对话\TET\train\outputs\egoschema-sub-sft2"
ANSWER_PATH = r"D:\work\实时对话\TET\outputs\qwenvl_test3\answer.jsonl"
OUTPUT_PATH = r"D:\work\实时对话\TET\train\outputs\egoschema-sub-sft_eval"
batch_size = 4

train_dataset, test_dataset, eval_dataset = generate_dataset(dataset_name, ANSWER_PATH)

# BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

config = PeftConfig.from_pretrained(model_id)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    config.base_model_name_or_path,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
)
model = PeftModel.from_pretrained(model, model_id)
processor = AutoProcessor.from_pretrained(model_id)
# Set padding side to left for decoder-only architecture
processor.tokenizer.padding_side = 'left'
acc = []
results = {}

# Process batches
test_dataset_batched = test_dataset.batch(batch_size)

for batch_idx, batch_data in enumerate(tqdm(test_dataset_batched)):
    try:
        # Apply chat template to all messages in batch
        texts = [
            processor.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True
            )
            for msg in batch_data["message"]
        ]

        # Ensure image token is in text for all samples
        image_token = processor.image_token
        texts = [
            text + f" {image_token}" if image_token not in text else text
            for text in texts
        ]

        # Process image/video inputs for all samples in batch
        batch_image_inputs = []
        batch_video_inputs = []
        for msg in batch_data["message"]:
            image_inputs, video_inputs = process_vision_info(msg)
            batch_image_inputs.append(image_inputs)
            batch_video_inputs.append(video_inputs)

        # Prepare inputs with padding
        inputs = processor(
            text=texts,
            videos=batch_video_inputs,
            return_tensors="pt",
            padding=True,  # Add padding for batch processing
        ).to("cuda")

        # Generate responses for the entire batch
        generated_ids = model.generate(
            **inputs, max_new_tokens=1024
        )  # Reduce token count

        # Trim the generated ids to only include the new tokens
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        # Decode all outputs in batch
        output_texts = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        # Process each output in the batch
        for i, (output_text, qid, truth) in enumerate(
            zip(output_texts, batch_data["qid"], batch_data["truth"])
        ):
            cur = {
                "raw": output_text,
                **format_output(output_text),
                "truth": truth,
            }
            results[qid] = cur
            acc.append(cur["answer"] == truth)
    except Exception as e:
        print(f"❌ Error at batch {batch_idx}: {e}")

output_file_path = os.path.join(OUTPUT_PATH, "results.json")
os.makedirs(OUTPUT_PATH, exist_ok=True)
with open(output_file_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print(f"Acc: {np.mean(acc):.2f}")
