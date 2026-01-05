import json
import sys
from pathlib import Path

import numpy as np
import os
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.resolve()))
from dataset.builder import build_dataset
from dataset.base import BaseDataset
from utils import get_video_length, get_video_size, load_data
from concurrent.futures import ThreadPoolExecutor, as_completed

from datasets import Dataset

from trl.trainer.utils import remove_none_values
from train_utils import compress_consecutive_numbers, concatenate_datasets
import random

RAW_OPTIONS = list("ABCDEF")

PROMPT = """
Based on the video, answer the following question: {question}. Locate the keyframes relevant to the question, answer it, and provide the reasoning process. Output a JSON with 3 keys: "keyframe", "reason", "answer". You can use "start-end" as a shorthand for keyframes. Do not add comments:
{{
    "keyframe": "keyframe related to the question, e.g. 1-90, 183, 185",
    "reason": "Provide a concise step-by-step analysis based on the visual information in the specified keyframes",
    "answer": "one from {options}"
}}
"""

PROMPT2 = """
Based on the video, answer the following question: {question}. Output only one character from {options}
"""

PROMPT3 = """
Based on the video, focus on the frame corresponding to {keyframe} and answer the following question: {question}. Output only one character from {options}.
"""

PROMPT4 = """
Based on the video, focus on the frame corresponding to {keyframe} and answer the following question: {question}. Output only one character from {options}. Please think step by step.
"""


def format_data(sample, fps, test=False, prompt_type="v1"):
    """
    Format a single dataset sample into the required structure.
    """
    options = "/".join(RAW_OPTIONS[: len(sample["options"])])
    format_data_kwargs = {"question": sample["question"], "options": options}
    prompt = {"v1": PROMPT, "v2": PROMPT2, "v3": PROMPT3}[prompt_type]
    out = {}
    if not test:
        assert "input_idx" in sample, "must provide input_idx for eval or train"
    if "input_idx" in sample:
        input_index = sorted(set(map(lambda idx: int(idx * fps), sample["input_idx"])))
        keyframe = compress_consecutive_numbers(input_index)
        if prompt_type == "v3":
            format_data_kwargs.update({"keyframe": keyframe})

        answer = {
            "reason": sample["explain"],
            "keyrfame": keyframe,
            "answer": sample["truth"],
        }
        out["keyframe"] = keyframe

    start_end = {k: sample[k] for k in ["video_start", "video_end"] if k in sample}
    message = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": sample["video_path"],
                    "max_pixels": 160 * 120,
                    "min_pixels": 0,
                    "fps": float(fps),
                    **start_end,
                },
                {
                    "type": "text",
                    "text": prompt.format(**format_data_kwargs),
                },
            ],
        },
    ]
    if not test:
        if prompt_type == "v1":
            gt_text = json.dumps(answer)
        else:
            gt_text = sample["truth"]
        message.append(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": gt_text,
                    }
                ],
            }
        )
    out.update(
        {
            "truth": sample["truth"],
            "qid": sample["qid"],
            "message": message,
        }
    )
    return out


def split_dataset(dataset, test_rate, eval_rate, seed=1234):
    """
    split list of data to (train, test, eval)
    """
    train_rate = 1 - test_rate - eval_rate
    assert train_rate <= 1, "test_rate + eval_rate must < 1"
    random.seed(seed)
    # 处理test_size：如果是比例，转换为整数
    total = len(dataset)
    test_size = int(total * test_rate)
    eval_size = int(total * eval_rate)
    train_size = total - test_size - eval_size

    # 生成索引并洗牌
    indices = list(range(total))
    random.shuffle(indices)  # 基于固定种子洗牌

    # 分割索引并提取对应元素
    test_idx = indices[:test_size]
    train_idx = indices[test_size : test_size + train_size]
    eval_idx = indices[test_size + train_size :]

    train_data = [dataset[idx] for idx in train_idx]
    test_data = [dataset[idx] for idx in test_idx]
    eval_data = [dataset[idx] for idx in eval_idx]

    return train_data, test_data, eval_data


def format_woker(sample, prompt_type, test=False, max_frame=100):
    size = get_video_length(sample["video_path"])
    fps = min(1, max_frame / size)
    return format_data(sample, fps, test, prompt_type), fps


def generate_dataset(
    dataset_cfg,
    dataset_config="./configs/dataset.yml",
    prompt_type="v1",
    max_frame=80,
    filter=None,
    split_test=False,
):
    """return (train dataset, test dataset, eval dataset)"""
    train_dataset = {}
    eval_dataset = {}
    test_dataset = {}

    for cfg in dataset_cfg:
        dataset_name = cfg["name"]
        answer_path = cfg.get("answer_path", None)
        test_rate = cfg["test_rate"]
        eval_rate = cfg["eval_rate"]
        dataset = build_dataset(
            dataset_config=dataset_config, name=dataset_name, is_training=False
        )
        if answer_path is not None:
            data_list = load_data(answer_path)
        else:
            assert test_rate == 1, "Eval or train dataset must provide answer_path"
        out = []
        for raw_data in dataset:
            qid = raw_data["qid"]
            sample = {}
            if answer_path is not None:
                data = data_list.get(qid, None)
                if data is None:
                    continue
                if filter is not None and not filter(data):
                    continue
                sample.update(data)
            video_path = os.path.join(dataset.config.video_path, raw_data["video_path"])
            sample = {**sample, **raw_data, "video_path": video_path}
            out.append(sample)

        train, test, eval = split_dataset(out, test_rate=test_rate, eval_rate=eval_rate)

        def run_task(data, desc, test):
            out_list, fps_list = [], []
            with ThreadPoolExecutor(max_workers=8) as executor:

                futures = [
                    executor.submit(format_woker, sample, prompt_type, test, max_frame)
                    for sample in data
                ]
                for future in tqdm(futures, total=len(data), desc=desc):
                    try:
                        out, fps = future.result()
                        out_list.append(out)
                        fps_list.append(fps)
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
            return out_list, fps_list

        for name, data, result in zip(
            ["train", "eval", "test"],
            [train, eval, test],
            [train_dataset, eval_dataset, test_dataset],
        ):
            if split_test and name != "test":
                continue
            if not split_test and name == "test":
                continue
            data, out_fps = run_task(
                data, f"format {dataset_name}[{name}]", name == "test"
            )
            print(f"{dataset_name}[{name}] fps: {np.mean(out_fps):.2f}")
            result[dataset_name] = data

    num_list = [
        sum([len(v) for v in v.values()])
        for v in [train_dataset, eval_dataset, test_dataset]
    ]
    for name, dataset, num in zip(
        ["train", "eval", "test"], [train_dataset, eval_dataset, test_dataset], num_list
    ):
        if split_test and name != "test":
            continue
        if not split_test and name == "test":
            continue
        print(f"{name} dataset: ")
        for k, v in dataset.items():
            print(f"{k}: {len(v)/num*100:.2f}%[{len(v)}/{num}]")

    def list2dataset(data):
        return Dataset.from_list(list(data)).with_transform(remove_none_values)

    if not split_test:
        train_dataset = list2dataset([d for v in train_dataset.values() for d in v])
        eval_dataset = list2dataset([d for v in eval_dataset.values() for d in v])
        return train_dataset, eval_dataset
    else:
        test_dataset = {key: list2dataset(value) for key, value in test_dataset.items()}
        return test_dataset


def format_output(output):
    try:
        out = json.loads(output)
    except json.JSONDecodeError:
        try:
            out = output.split("{")[1].split("}")[0]
            out = json.loads(f"{{{out}}}")
        except Exception as e:
            print(f"failed to decode json: {output}")
            out = None
    return out


def parse_multi_choice_response(data):
    return BaseDataset.parse_multi_choice_response(data)
