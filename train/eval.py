import tokenize
import torch
import os
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from data_utils import (
    Prompt,
    generate_dataset,
    parse_multi_choice_response,
    JSONLStorage,
    prepare_inputs,
)

from transformers import BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import numpy as np
from tqdm import tqdm
from utils import load_data, save_data

# Model and processor setup
# model_id = r"D:\work\实时对话\TET\train\outputs\egoschema-sub-sft3\checkpoint-4100"
# model_id = r"D:\work\实时对话\TET\train\outputs\egoschema-sub-sft4\checkpoint-700"
# model_id = r"D:\work\实时对话\TET\train\outputs\sft6\checkpoint-4700"
model_id = r"D:\work\实时对话\TET\train\outputs\sft8_2\checkpoint-3000"
# model_id = r"D:\models\Video-R1-7B"

data_cfg_path = r"D:\work\实时对话\TET\train\config\dataset_cfg.yml"  # for answer2
# data_cfg_path = r"D:\work\实时对话\TET\train\config\dataset_cfg_eval1.yml" # for answer1
# data_cfg_path = r"D:\work\实时对话\TET\train\config\dataset_cfg_eval2.yml" # for answer2
data_cfg = load_data(data_cfg_path)

R1_MODEL = False
TEST_SFT = True

batch_size = 8
# PROMPT_TYPE = "v1"  # 推理增强
PROMPT_TYPE = "v1_5"  # 推理增强
# PROMPT_TYPE = "v2"  # 直接输出答案
# PROMPT_TYPE = "v3"  # 让模型关注关键帧
# PROMPT_TYPE = "r1"

OUTPUT_PATH = (
    f"{model_id}_p{PROMPT_TYPE}{'_sft' if TEST_SFT else ''}_eval"  # for answer2
)
# OUTPUT_PATH = f"{model_id}_p{PROMPT_TYPE}{'_sft' if TEST_SFT else ''}_eval2" # for answer1
# OUTPUT_PATH = f"{model_id}_p{PROMPT_TYPE}{'_sft' if TEST_SFT else ''}_eval2" # for answer2

prompt = Prompt.create(PROMPT_TYPE)
# data_cfg = load_data(data_cfg_path)
test_dataset = generate_dataset(
    data_cfg,
    prompt=prompt,
    split_test=True,
)
save_data(data_cfg, os.path.join(OUTPUT_PATH, "data_cfg.yml"))

# BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

processor = AutoProcessor.from_pretrained(model_id)
# Set padding side to left for decoder-only architecture
processor.tokenizer.padding_side = "left"

if R1_MODEL:
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        use_cache=True,
    )

else:
    config = PeftConfig.from_pretrained(model_id)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.base_model_name_or_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
    )
    if len(processor.tokenizer) != model.get_output_embeddings().out_features:
        model.resize_token_embeddings(len(processor.tokenizer))
    if TEST_SFT:
        model = PeftModel.from_pretrained(model, model_id)


def calc_acc(sample):
    return parse_multi_choice_response(sample["answer"]) == sample["truth"]


for name, dataset in test_dataset.items():
    results_path = os.path.join(OUTPUT_PATH, f"result_{name}.jsonl")
    storage = JSONLStorage(results_path, "qid")

    acc = {}

    def filter(sample):
        if not storage.has(sample["qid"]):
            return True
        if "error" in storage.get(sample["qid"]):
            storage.delete(sample["qid"])
            return True
        sample = storage.get(sample["qid"])
        acc[sample["qid"]] = calc_acc(sample)
        return False

    # Process batches
    test_dataset_batched = dataset.filter(filter).batch(batch_size)
    storage.write_all()
    for batch_idx, batch_data in enumerate(
        tqdm(test_dataset_batched, desc=f"test {name}")
    ):
        try:
            inputs = prepare_inputs(
                processor, batch_data, add_generation_prompt=True
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
                    output = prompt.format_output(output_text)
                    cur = {
                        **output,
                        "truth": truth,
                        "qid": qid,
                    }
                    if "keyframe" in batch_data:
                        cur["gt_keyframe"] = batch_data["keyframe"][i]
                    storage.write(cur)
                    acc[qid] = calc_acc(cur)
                except Exception as e:
                    storage.write({"qid": qid, "output": output_text, "error": str(e)})
                    continue
        except Exception as e:
            import traceback

            traceback.print_exc()
    acc = list(acc.values())
    acc = np.array(acc)
    acc_value = acc.mean() if len(acc) != 0 else 0
    print(f"{name} Acc: {acc_value*100:.2f}[{acc.sum()}/{len(acc)}]")
