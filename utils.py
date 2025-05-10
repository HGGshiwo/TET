import json
import yaml
from pathlib import Path
import base64, io
import re
import numpy as np
import torch
from typing import List, Generator
import math
import torch.nn as nn
import jsonlines
from PIL import Image

def save_data(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".json":
        json.dump(data, path.open("w"), indent=4)
    elif path.suffix == ".yml":
        yaml.dump(data, path.open("w"), indent=4)
    else:
        path.write_text(data)

def load_data(path):
    path = Path(path)
    if path.suffix == ".json":
        return json.load(path.open(encoding="utf-8"))
    elif path.suffix == ".yml":
        return yaml.safe_load(path.open(encoding="utf-8"))
    elif path.suffix == ".jsonl":
        out = {}
        for value in jsonlines.open(path, "r"):
            if "qid" in value:
                key = value["qid"]
            elif "vid" in value:
                key = value["vid"]
            else:
                raise ValueError(f"Invalid key in jsonl file: {value}")
            if "idx" in value:
                if key not in out:
                    out[key] = {}
                out[key][value['idx']] = value
            else:
                out[key] = value
        return out
    else:
        return path.read_text()


def chunk(arr, chunk_size):
    if isinstance(arr, Generator):
        def gen_chunk():
            chunked = []
            while True:
                try:
                    chunked.append(next(arr))
                except StopIteration:
                    if len(chunked) > 0:
                        yield chunked
                    return
                if len(chunked) == chunk_size:
                    yield chunked
                    chunked = []
        return gen_chunk()
    else:
        return [arr[i : i + chunk_size] for i in range(0, len(arr), chunk_size)]

def parse_relevance(text):
    try:
        text = int(text)
    except:
        text = re.findall(r"\d+", text)
        text = text[0] if len(text) > 0 else None
    try:
        score = int(text)
        assert score in [1, 2, 3]
    except:
        score = 1
    return score
    

def image2base64(frame, img_type="JPEG"):
    if isinstance(frame, torch.Tensor):
        frame = frame.cpu().numpy()
    if isinstance(frame, np.ndarray):
        frame = Image.fromarray(frame)
    buffer = io.BytesIO()
    frame.save(buffer, format=img_type)
    image_bytes = buffer.getvalue()
    img_b64_str = base64.b64encode(image_bytes).decode("utf-8")
    mime_type = f"image/{img_type.lower()}"
    return f"data:{mime_type};base64,{img_b64_str}"


def pad(batch: List[List], pad_value=-100, pdding_side="left"):
    max_length = max(len(x) for x in batch)
    pad_batch = []
    masks = []
    for data in batch:
        pad_length = max_length - len(data)
        if pdding_side == "left":
            pad_data = data + [pad_value] * pad_length
            mask = [1] * len(data) + [0] * pad_length
        else:
            pad_data = [pad_value] * pad_length + data
            mask = [0] * pad_length + [1] * len(data)
        pad_batch.append(pad_data)
        masks.append(mask)
    return torch.as_tensor(pad_batch), torch.as_tensor(masks)


def load_frame_features(name_ids, save_folder):
    filename = f"{name_ids}.pt"  # Construct filename with name_ids
    filepath = Path(save_folder).joinpath(filename)
    img_feats = torch.load(filepath, weights_only=True, map_location="cpu")
    return img_feats


def uniform_sample(arr, k):
    # Use linspace to generate k evenly spaced indices, then round and convert to integers.
    indices = np.linspace(0, len(arr) - 1, k, dtype=int)
    indices = list(set(indices))
    return [arr[i] for i in indices]


def pool2d(image_feature, stride=2, mm_spatial_pool_mode="average"):
    height = width = int(math.sqrt(image_feature.shape[1]))
    num_frames, num_tokens, num_dim = image_feature.shape
    image_feature = image_feature.view(num_frames, height, width, -1)
    image_feature = image_feature.permute(0, 3, 1, 2).contiguous()
    # image_feature = nn.functional.max_pool2d(image_feature, self.config.mm_spatial_pool_stride)
    if mm_spatial_pool_mode == "average":
        image_feature = nn.functional.avg_pool2d(image_feature, stride)
    elif mm_spatial_pool_mode == "max":
        image_feature = nn.functional.max_pool2d(image_feature, stride)
    elif mm_spatial_pool_mode == "bilinear":
        height, width = image_feature.shape[2:]
        scaled_shape = [math.ceil(height / stride), math.ceil(width / stride)]
        image_feature = nn.functional.interpolate(
            image_feature, size=scaled_shape, mode="bilinear"
        )

    else:
        raise ValueError(f"Unexpected mm_spatial_pool_mode: {mm_spatial_pool_mode}")
    image_feature = image_feature.permute(0, 2, 3, 1)
    image_feature = image_feature.view(num_frames, -1, num_dim)
    return image_feature


def add_token_per_grid(image_feature, image_newline):
    resize_h = int(math.sqrt(image_feature.shape[1]))
    num_frames = image_feature.shape[0]
    feature_dim = image_feature.shape[-1]

    image_feature = image_feature.view(num_frames, 1, resize_h, resize_h, -1)
    image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
    image_feature = image_feature.flatten(1, 2).flatten(2, 3)
    image_feature = torch.cat(
        (
            image_feature,
            image_newline[:, None, None]
            .expand(*image_feature.shape[:-1], 1)
            .to(image_feature.device),
        ),
        dim=-1,
    )
    image_feature = image_feature.flatten(1, 2).transpose(0, 1)
    return image_feature


def get_token_per_frame(image_feature, mm_spatial_pool_stride, mm_spatial_pool_mode):
    B, N, F = image_feature.shape
    image_feature = pool2d(
        torch.zeros(1, N, F), mm_spatial_pool_stride, mm_spatial_pool_mode
    )
    image_newline = torch.zeros(F)
    token_per_frame = add_token_per_grid(image_feature, image_newline).shape[0]
    return token_per_frame
