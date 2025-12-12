import jsonlines
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
from utils import load_data, save_data

# Model and processor setup
model_id = r"D:\work\实时对话\TET\train\outputs\egoschema-sub-sft"

# data_cfg_path = r"D:\work\实时对话\TET\train\config\dataset_cfg_eval1.yml" # for answer1
data_cfg_path = r"D:\work\实时对话\TET\train\config\dataset_cfg_eval2.yml" # for answer2
data_cfg = load_data(data_cfg_path)
TEST_SFT = False
batch_size = 8
# PROMPT_TYPE = "v1"  # 推理增强
# PROMPT_TYPE = "v2" # 直接输出答案
PROMPT_TYPE = "v3"  # 让模型关注关键帧

# OUTPUT_PATH = f"{model_id}_p{PROMPT_TYPE}{'_sft' if TEST_SFT else ''}_eval2" # for answer1
OUTPUT_PATH = f"{model_id}_p{PROMPT_TYPE}{'_sft' if TEST_SFT else ''}_eval2" # for answer2

# data_cfg = load_data(data_cfg_path)
train_dataset, test_dataset, eval_dataset = generate_dataset(
    data_cfg, prompt_type=PROMPT_TYPE
)
save_data(data_cfg, os.path.join(OUTPUT_PATH, "data_cfg.yml"))

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
if TEST_SFT:
    model = PeftModel.from_pretrained(model, model_id)
processor = AutoProcessor.from_pretrained(model_id)
# Set padding side to left for decoder-only architecture
processor.tokenizer.padding_side = "left"

for name, dataset in test_dataset.items():
    results_path = os.path.join(OUTPUT_PATH, f"result_{name}.jsonl")
    results = load_data(results_path) if os.path.exists(results_path) else {}
    writer = jsonlines.open(results_path, "a")

    acc = {}

    def filter(sample):
        if sample["qid"] not in results:
            return True
        sample = results[sample["qid"]]
        acc[sample["qid"]] = sample["truth"] == sample["answer"].strip()[0]
        return False

    # Process batches
    test_dataset_batched = dataset.filter(filter).batch(batch_size)
    for batch_idx, batch_data in enumerate(
        tqdm(test_dataset_batched, desc=f"test {name}")
    ):
        try:
            # Apply chat template to all messages in batch
            texts = [
                processor.apply_chat_template(
                    msg, tokenize=False, add_generation_prompt=True
                )
                for msg in batch_data["message"]
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
                **inputs,
                do_sample=False,  # 关闭采样
                num_beams=1,  # 使用贪婪搜索（beam=1）
                max_new_tokens=1024,
            )
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
                try:
                    if PROMPT_TYPE == "v1":
                        output = format_output(output_text)
                    else:
                        output = {"answer": output_text}
                    cur = {
                        "raw": output_text,
                        **output,
                        "truth": truth,
                        "qid": qid,
                    }
                    writer.write(cur)
                    acc[qid] = cur["answer"].strip()[0] == truth
                except Exception as e:
                    print(f"❌ Error at {output_text}: {e}")
                    continue
        except Exception as e:
            print(f"❌ Error at batch {batch_idx}: {e}")
    acc = list(acc.values())
    acc = np.array(acc)
    acc_value = acc.mean() if len(acc) != 0 else 0
    print(f"{name} Acc: {acc_value*100:.2f}[{acc.sum()}/{len(acc)}]")
