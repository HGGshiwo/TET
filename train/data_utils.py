from shlex import join
from typing import Callable, Optional, Union, Tuple, List, Any, Dict
import time


import torch
from qwen_vl_utils.vision_process import (
    VIDEO_READER_BACKENDS,
    calculate_video_frame_range,
    smart_nframes,
)
from transformers import AutoProcessor


def _read_video_decord(
    ele: Dict[str, Any],
) -> Tuple[torch.Tensor, float]:
    """修改原始的_read_video_decord, .asnumpy()缺失的问题"""
    import decord

    video_path = ele["video"]
    st = time.time()
    vr = decord.VideoReader(video_path)
    total_frames, video_fps = len(vr), vr.get_avg_fps()
    start_frame, end_frame, total_frames = calculate_video_frame_range(
        ele,
        total_frames,
        video_fps,
    )
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    idx = torch.linspace(start_frame, end_frame, nframes).round().long().tolist()
    video = vr.get_batch(idx)
    video = video.permute(0, 3, 1, 2)  # Convert to TCHW format
    # logger.info(
    #     f"decord:  {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s"
    # )
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps

    video_metadata = dict(
        fps=video_fps,
        frames_indices=idx,
        total_num_frames=total_frames,
        video_backend="decord",
    )
    return video, video_metadata, sample_fps


VIDEO_READER_BACKENDS["decord"] = _read_video_decord
from qwen_vl_utils import process_vision_info as _process_vision_info


def process_vision_info(
    conversations: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]],
    return_video_kwargs: bool = False,
    return_video_metadata: bool = False,
    image_patch_size: int = 14,
):
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for key in ["video_start", "video_end", "max_frames", "min_frames"]:
            if np.isnan(conversation[0]["content"][0][key]):
                del conversation[0]["content"][0][key]
    return _process_vision_info(
        conversations, return_video_kwargs, return_video_metadata, image_patch_size
    )


from functools import partial
import json
import sys
from pathlib import Path

import jsonlines
import numpy as np
import os
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.resolve()))
from dataset.builder import build_dataset
from dataset.base import BaseDataset
from utils import get_video_length, get_video_size, load_data, save_data
from concurrent.futures import ThreadPoolExecutor, as_completed
import inspect
from datasets import Dataset

from trl.trainer.utils import remove_none_values
from train_utils import compress_consecutive_numbers, concatenate_datasets
import random
import textwrap
import re

RAW_OPTIONS = list("ABCDEF")


class Prompt:
    prompt_map = {}

    def __init_subclass__(cls):
        cls.prompt_map[cls.version] = cls

    @classmethod
    def get_special_tokens(cls):
        """需要添加的special tokens"""
        return []

    @classmethod
    def create(cls, version: str) -> "Prompt":
        return cls.prompt_map[version]()

    @classmethod
    def get_keys(cls):
        func_signature = inspect.signature(cls.build)
        param_names = [param.name for param in func_signature.parameters.values()]
        return param_names

    def strip(data: str):
        return textwrap.dedent(data).strip()

    @classmethod
    def get_gt(cls, item: dict):
        """构造ground truth"""
        return item["answer"]

    @classmethod
    def format_output(cls, output: str):
        """解析模型输出"""
        return {"answer": output}


class PromptV1(Prompt):
    version = "v1"

    @classmethod
    def build(cls, question, options):
        PROMPT = """
        Based on the video, answer the following question: {question}. Locate the keyframes relevant to the question, answer it, and provide the reasoning process. Output a JSON with 3 keys: "reasoning", "keyframe",  "answer". Do not add comments:
        {{
            "reasoning": "Provide a list of concise step-by-step analysis based on the visual information in the specified keyframes",
            "keyframes": "keyframe related to the question",
            "answer": "one from {options}"
        }}
        """
        return cls.strip(PROMPT).format(question=question, options=options)

    @classmethod
    def format_output(cls, output):
        try:
            out = json.loads(output)
        except json.JSONDecodeError:
            try:
                out = output.split("{")[1].split("}")[0]
                out = json.loads(f"{{{out}}}")
            except Exception as e:
                raise ValueError(f"failed to decode json")
                # out = None
        return out

    @classmethod
    def get_gt(cls, item: dict):
        return json.dumps(item)


class PromptV1_5(Prompt):
    version = "v1_5"
    split_token = "<split>"
    step_split_token = "<step_split>"

    @classmethod
    def get_special_tokens(cls):
        return [cls.split_token, cls.step_split_token]

    @classmethod
    def build(cls, question, options):
        PROMPT = """
        Based on the video, answer the question: {question}
        **Output Rules**:
        1. Split Reasoning, Keyframe, Answer with {split_token}
        2. Reasoning: Step-by-step analysis, separated by {step_split_token}
        3. Keyframe: Frame/timestamp related to the question
        4. Answer: EXACTLY one option from {options}

        **Output Example**:
        Reasoning step1{step_split_token}step2{split_token}Keyframe: 15,17,20,23{split_token}Answer: [Option from {options}]
        """
        return cls.strip(PROMPT).format(
            question=question,
            options=options,
            step_split_token=cls.step_split_token,
            split_token=cls.split_token,
        )

    @classmethod
    def format_output(cls, output: str):
        out = {}
        split_out = output.split(cls.split_token)
        if len(split_out) != 3:
            out = {"error": "num of output is not equal to 3", "raw": output}
        else:
            out["reasoning"] = split_out[0].split(cls.step_split_token)
            out["keyframes"] = split_out[1]
            out["answer"] = split_out[2]
        return out

    @classmethod
    def get_gt(cls, item: dict):
        reasoning = cls.step_split_token.join(item["reasoning"])
        out = cls.split_token.join([reasoning, item["keyframes"], item["answer"]])
        return out


class PromptV2(Prompt):
    version = "v2"

    @classmethod
    def build(cls, question, options):
        PROMPT2 = """
        Based on the video, answer the following question: {question}. Output only one character from {options}
        """
        return cls.strip(PROMPT2).format(question=question, options=options)


class PromptV3(Prompt):
    version = "v3"

    @classmethod
    def build(cls, question, options, keframe):
        PROMPT3 = """
        Based on the video, focus on the frame corresponding to {keyframe} and answer the following question: {question}. Output only one character from {options}.
        """
        return cls.srtip(PROMPT3).format(question=question, options=options)


class PromptR1(Prompt):
    version = "r1"

    @classmethod
    def build(cls, question, options):
        QUESTION_TEMPLATE = (
            "{question}\n{options}"
            "Please think about this question as if you were a human pondering deeply. "
            "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions "
            "It's encouraged to include self-reflection or verification in the reasoning process. "
            "Provide your detailed reasoning between the <think> and </think> tags, and then give your final answer between the <answer> and </answer> tags."
            "Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags."
        )
        return cls.strip(QUESTION_TEMPLATE).format(question=question, options=options)

    @classmethod
    def format_output(cls, output: str):
        out = output.split("<answer>")
        return {"reasoning": out[0], "answer": out[1].replace("</answer>", "")}


def format_data(sample, prompt: Prompt, test=False):
    """
    Format a single dataset sample into the required structure.
    """
    options = "/".join(RAW_OPTIONS[: len(sample["options"])])
    format_data_kwargs = {"question": sample["question"], "options": options}
    out = {}
    if not test:
        assert "reasoning" in sample, "must provide reasoning for eval or train"

    if "reasoning" in sample:
        input_index = sample["keyframes"]
        keyframe = compress_consecutive_numbers(input_index)
        if "keyframe" in prompt.get_keys():
            format_data_kwargs.update({"keyframe": keyframe})

        answer = {
            "reasoning": sample["reasoning"],
            "keyframes": keyframe,
            "answer": sample["truth"],
        }
        out["keyframe"] = keyframe

    message = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": sample["video_path"],
                    "max_pixels": 160 * 120,
                    "min_pixels": 0,
                    "video_start": sample.get("video_start", np.nan),
                    "video_end": sample.get("video_end", np.nan),
                    "max_frames": sample["max_frames"],
                    "min_frames": sample["min_frames"],
                    "fps": sample["fps"],
                },
                {
                    "type": "text",
                    "text": prompt.build(**format_data_kwargs),
                },
            ],
        },
    ]
    if not test:
        gt_text = prompt.get_gt(answer)
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


def format_woker(sample: Dict[str, Any], prompt: Prompt, test: bool = False):
    return format_data(sample, prompt, test)


def generate_dataset(
    dataset_cfg: Dict[Any, str],
    prompt: Prompt,
    dataset_config: str = "./configs/dataset.yml",
    filter: Optional[Callable] = None,
    split_test: Optional[bool] = False,
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
        fps = cfg.get("fps", 1)
        min_frames = cfg.get("min_frames", np.nan)
        max_frames = cfg.get("max_frames", np.nan)
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
            sample = {
                **sample,
                **raw_data,
                "video_path": video_path,
                "fps": fps,
                "min_frames": min_frames,
                "max_frames": max_frames,
            }
            out.append(sample)

        train, test, eval = split_dataset(out, test_rate=test_rate, eval_rate=eval_rate)

        def run_task(data, desc, test):
            out_list, fps_list = [], []
            if len(data) == 0:
                return out_list
            with ThreadPoolExecutor(max_workers=8) as executor:

                futures = [
                    executor.submit(format_woker, sample, prompt, test)
                    for sample in data
                ]
                for future in tqdm(futures, total=len(data), desc=desc):
                    try:
                        out = future.result()
                        out_list.append(out)
                    except Exception as e:
                        import traceback

                        traceback.print_exc()
            return out_list

        for name, data, result in zip(
            ["train", "eval", "test"],
            [train, eval, test],
            [train_dataset, eval_dataset, test_dataset],
        ):
            if split_test and name != "test":
                continue
            if not split_test and name == "test":
                continue
            data = run_task(data, f"format {dataset_name}[{name}]", name == "test")
            # print(f"{dataset_name}[{name}] fps: {np.mean(out_fps):.2f}")
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
        dataset = Dataset.from_list(list(data)).with_transform(remove_none_values)
        return dataset

    if not split_test:
        train_dataset = list2dataset([d for v in train_dataset.values() for d in v])
        eval_dataset = list2dataset([d for v in eval_dataset.values() for d in v])
        return train_dataset, eval_dataset
    else:
        test_dataset = {key: list2dataset(value) for key, value in test_dataset.items()}
        return test_dataset


def format_output(output: str) -> Dict[str, Any]:
    try:
        out = json.loads(output)
    except json.JSONDecodeError:
        try:
            out = output.split("{")[1].split("}")[0]
            out = json.loads(f"{{{out}}}")
        except Exception as e:
            raise ValueError(f"failed to decode json")
            # out = None
    return out


def parse_multi_choice_response(data):
    return BaseDataset.parse_multi_choice_response(data)


def prepare_inputs(
    processor: AutoProcessor, batch_data: dict | list, add_generation_prompt
):
    # Apply chat template to all messages in batch
    if isinstance(batch_data, list):
        message = [data["message"] for data in batch_data]
    else:  # 如果是dict，直接取key
        message = batch_data["message"]

    texts = [
        processor.apply_chat_template(
            msg, tokenize=False, add_generation_prompt=add_generation_prompt
        )
        for msg in message
    ]

    # Process image/video inputs for all samples in batch
    batch_image_inputs = []
    batch_video_inputs = []
    for msg in message:
        image_inputs, video_inputs = process_vision_info(msg)
        batch_image_inputs.append(image_inputs)
        batch_video_inputs.append(video_inputs)

    # Prepare inputs with padding
    inputs = processor(
        text=texts,
        videos=batch_video_inputs,
        return_tensors="pt",
        padding=True,  # Add padding for batch processing
    )

    return inputs


class Storage:
    def __init__(self, path, idx_key, filter=None):
        self.path = path
        self.idx_key = idx_key
        self.filter = filter
        self.data = {}

    def has(self, key):
        return key in self.data

    def write(self, **kwargs):
        pass

    def __del__(self):
        if hasattr(self, "wirter"):
            self.writer.close()


class JSONLStorage(Storage):
    def __init__(self, path, idx_key, filter=None):
        super().__init__(path, idx_key, filter)
        self.data = {}
        self.writer = jsonlines.open(path, "a")
        if os.path.exists(path):
            self.data = load_data(path)
        self.path = path

    def has(self, key):
        return key in self.data

    def get(self, key):
        return self.data[key]

    def write(self, data):
        self.writer.write(data)

    def delete(self, key):
        del self.data[key]

    def write_all(self):
        self.writer.close()
        with jsonlines.open(self.path, "w") as f:
            f.write_all(self.data.values())
        self.writer = jsonlines.open(self.path, "a")
