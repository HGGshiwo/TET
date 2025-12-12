import json
import sys
from pathlib import Path

from json5 import JSON5Encoder

sys.path.append(str(Path(__file__).parent.parent.resolve()))
from dataset.builder import build_dataset
from utils import load_data
from tqdm import tqdm
from datasets import Dataset, concatenate_datasets

from trl.trainer.utils import remove_none_values
import re
from train_utils import compress_consecutive_numbers

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


def format_data(sample, fps=1, test=False, prompt_type="v1"):
    """
    Format a single dataset sample into the required structure.
    """
    assert fps <= 1, "fps must <= 1"
    input_index = sorted(set(map(lambda idx: int(idx * fps), sample["input_idx"])))
    keyframe = compress_consecutive_numbers(input_index)
    prompt = {"v1": PROMPT, "v2": PROMPT2, "v3": PROMPT3}[prompt_type]
    options = "/".join(RAW_OPTIONS[: len(sample["options"])])
    format_data = {"question": sample["question"], "options": options}
    if prompt_type == "v3":
        format_data.update({"keyframe": keyframe})
    answer = {
        "reason": sample["explain"],
        "keyrfame": keyframe,
        "answer": sample["truth"],
    }
    message = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": sample["video_path"],
                    "max_pixels": 320 * 240,
                    "min_pixels": 0,
                    "fps": fps,
                },
                {
                    "type": "text",
                    "text": prompt.format(**format_data),
                },
            ],
        },
    ]
    if not test:
        message.append(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(answer),
                    }
                ],
            }
        )
    return {
        "truth": sample["truth"],
        "qid": sample["qid"],
        "message": message,
        "keyframe": keyframe,
    }


def split_dataset(
    dataset, test_rate, eval_rate, *, test_map_kwargs=None, train_map_kwargs=None
):
    """ test_rate == 1: test
        test_rate < 1 + eval_rate == 0: test + train
        test_rate < 1 + eval_rate > 0 : test + train + eval
    """
    train_eval_rate = 1 - test_rate
    train_rate = 1 - test_rate - eval_rate
    assert train_rate <= 1, "test_rate + eval_rate must < 1"

    if train_rate != 0:
        train_rate = train_rate / (train_rate + eval_rate)
        eval_rate = 1 - train_rate
        train_test_dataset = dataset.train_test_split(test_size=test_rate, seed=1234)
        train_eval_dataset, test_dataset = (
            train_test_dataset["train"],
            train_test_dataset["test"],
        )
    else:
        train_eval_dataset = Dataset.from_list([])
        test_dataset = dataset
    if train_map_kwargs is not None:
        train_eval_dataset = train_eval_dataset.map(**train_map_kwargs)
    if test_map_kwargs is not None:
        test_dataset = test_dataset.map(**test_map_kwargs)
    if eval_rate != 0:
        train_eval_dataset = train_eval_dataset.train_test_split(
            test_size=eval_rate, seed=1234
        )
        train_dataset = train_eval_dataset["train"]
        eval_dataset = train_eval_dataset["test"]
    else:
        train_dataset = train_eval_dataset
        eval_dataset = Dataset.from_list([])
    return train_dataset, eval_dataset, test_dataset


def generate_dataset(
    dataset_cfg,
    dataset_config="./configs/dataset.yml",
    prompt_type="v1",
    filter=None,
):
    train_dataset = {}
    eval_dataset = {}
    test_dataset = {}

    for cfg in dataset_cfg:
        dataset_name = cfg["name"]
        answer_path = cfg["answer_path"]
        fps = cfg["fps"]
        test_rate = cfg["test_rate"]
        eval_rate = cfg["eval_rate"]
        dataset = build_dataset(
            dataset_config=dataset_config, name=dataset_name, is_training=False
        )
        data_list = load_data(answer_path)
        out = []
        for raw_data in dataset:
            qid = raw_data["qid"]
            data = data_list.get(qid, None)
            if data is None:
                continue
            if filter is not None and not filter(data):
                continue
            video_path = str(
                Path(dataset.config.video_path).joinpath(raw_data["video_path"])
            )
            sample = {**data, **raw_data, "video_path": video_path}
            out.append(sample)

        dataset = Dataset.from_list(out)
        dataset = dataset.with_transform(remove_none_values)
        format_kwargs = {"fps": fps, "prompt_type": prompt_type}

        train, eval, test = split_dataset(
            dataset,
            test_rate=test_rate,
            eval_rate=eval_rate,
            train_map_kwargs=dict(
                function=lambda sample: format_data(sample, **format_kwargs),
                desc=f"format {dataset_name}[train+eval]",
            ),
            test_map_kwargs=dict(
                function=lambda sample: format_data(sample, test=True, **format_kwargs),
                desc=f"format {dataset_name}[test]",
            ),
        )
        train_dataset[dataset_name] = train
        test_dataset[dataset_name] = test
        eval_dataset[dataset_name] = eval
    train_dataset = concatenate_datasets(list(train_dataset.values()))
    eval_dataset = concatenate_datasets(list(eval_dataset.values()))
    return train_dataset, test_dataset, eval_dataset


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
