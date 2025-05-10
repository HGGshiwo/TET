from transformers import PreTrainedTokenizer
from collections import defaultdict
from utils import *
from functools import partial
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
)
from llava.mm_utils import (
    tokenizer_image_token,
)
from llava.conversation import conv_templates
import copy
from PIL import Image

prompt1 = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n[question]"
prompt2 = "<|im_end|>\n<|im_start|>assistant\n"


def long_video_data_collator(
    batch, tokenizer, mm_spatial_pool_stride, mm_spatial_pool_mode, is_training=True
):
    outputs = defaultdict(list)
    token_per_frame = None
    assistant_ids = tokenizer.assistant_ids
    for data in batch:
        conversation = prompt1.replace("[question]", data["question"])
        conversation_ids = tokenizer(conversation, add_special_tokens=False).input_ids
        frame_features = load_frame_features(data["vid"], data["feature_path"])
        if token_per_frame is None:
            token_per_frame = get_token_per_frame(
                frame_features, mm_spatial_pool_stride, mm_spatial_pool_mode
            )
        v_placeholder_id = tokenizer.v_placeholder_id
        offset = len(conversation_ids)

        truth = data["truth"]
        truth_ids = tokenizer(truth, add_special_tokens=False).input_ids
        if "primary_cluster_ids" in data:
            conversation_ids += (
                len(data["primary_cluster_ids"]) * token_per_frame * [v_placeholder_id]
            )
        if is_training:
            conversation_ids += (
                len(data["sub_cluster_ids"]) * token_per_frame * [v_placeholder_id]
            )
            labels = (
                [-100] * len(conversation_ids + assistant_ids)
                + truth_ids
                + [tokenizer.eos_token_id]
            )
            input_ids = (
                conversation_ids + assistant_ids + [tokenizer.bos_token_id] + truth_ids
            )
            rel_mask = []
            # If long video
            if "sub_cluster_ids" in data:
                outputs["sub_cluster_ids"].append(data["sub_cluster_ids"])
                for i, relevance in enumerate(data["relevance"]):
                    rel_mask.append(offset + (i + 1) * token_per_frame - 1)
            outputs["labels"].append(labels)
        else:
            input_ids = conversation_ids
            outputs["labels"].append(truth_ids)
            
        outputs["rel_mask"].append(rel_mask)
        outputs["truth"].append(truth)  # for evaluation
        outputs["relevance"].append(data["relevance"])
        for key, value in data.items():
            if key.startswith("cm_"):
                outputs[key].append(value)  # for evaluation
        outputs["input_ids"].append(input_ids)
        if "primary_cluster_ids" in data:
            outputs["primary_cluster_ids"].append(data["primary_cluster_ids"])
        outputs["frame_features"].append(frame_features)

    # Padding labels and input_ids
    outputs["input_ids"], outputs["attention_mask"] = pad(outputs["input_ids"])
    if "labels" in outputs:
        outputs["labels"], _ = pad(outputs["labels"])
    outputs = dict(outputs)
    return outputs


def video_llava_data_collator(
    batch,
    tokenizer: PreTrainedTokenizer,
    image_processor=None,
    delay_load=True,
    select_type="uniform",
    frame_limit=64,
):
    conv_template = "qwen_1_5"
    outputs = defaultdict(list)
    for item in batch:
        if delay_load:
            feature = load_frame_features(item["vid"], item["feature_path"])
            num_frame = feature.shape[0]
        else:
            frame_path = Path(item["frame_path"])
            image_paths = list(frame_path.joinpath(item["vid"]).iterdir())
            num_frame = len(image_paths)
        select_idx = range(0, num_frame)
        if "time" in item:
            select_idx = select_idx[: item["time"]]
        if "primary_cluster_ids" in item:
            select_idx = [idx for idx in item["sub_cluster_ids"] if idx in select_idx]
            select_idx = sorted(list(select_idx))
            select_num = min(
                frame_limit, len(select_idx)
            )  # video_llava support max 64 frames
            select_num = frame_limit if select_num == 0 else select_num

            if select_type == "uniform" or len(select_idx) == 0:
                select_idx = uniform_sample(list(range(0, num_frame)), select_num)
            elif select_type == "relevance":
                if select_num != len(select_idx):
                    select_idx = uniform_sample(select_idx, select_num)
            else:
                raise ValueError(f"select_type {select_type} not supported")
        if delay_load:
            video = [feature[idx] for idx in select_idx]
            video = torch.stack(video).unsqueeze(-1)  # unsqueeze to image shape
        else:
            image_paths = [image_paths[idx] for idx in select_idx]
            video = [Image.open(path) for path in image_paths]
            video = image_processor.preprocess(video, return_tensors="pt")[
                "pixel_values"
            ]
        video = video.to(torch.bfloat16)
        question = DEFAULT_IMAGE_TOKEN + item["question"]
        labels = tokenizer(item["truth"]).input_ids
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX)

        outputs["input_ids"].append(input_ids)
        outputs["labels"].append(labels)
        outputs["images"].append(video)
        outputs["modalities"].append("video")
        outputs["truth"].append(item["truth"])
        
        for key, value in item.items():
            if key.startswith("cm_"):
                outputs[key].append(value)  # for evaluation
        

    outputs["input_ids"], outputs["attention_mask"] = pad(outputs["input_ids"])
    outputs["labels"], _ = pad(outputs["labels"])
    return dict(outputs)


def get_data_collator(model_type: str, **kwargs):
    data_collator = globals()[f"{model_type}_data_collator"]
    data_collator = partial(data_collator, **kwargs)
    return data_collator
