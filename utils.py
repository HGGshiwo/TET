import json
import yaml
from pathlib import Path
import base64, io
import re
import numpy as np
import torch
import clip
import math
import jsonlines
import decord
from PIL import Image, ImageDraw, ImageFont
from openai import AsyncOpenAI, AsyncAzureOpenAI
from dotenv import load_dotenv
from collections import defaultdict
from torchvision.utils import make_grid as tv_make_grid
from typing import List, Generator
import os
import requests
from qwen_vl_utils import smart_resize

decord.bridge.set_bridge("torch")

from tqdm import tqdm
import sys
import contextlib


class DummyFile:
    def __init__(self, file):
        if file is None:
            file = sys.stderr
        self.file = file

    def write(self, x):
        if len(x.rstrip()) > 0:
            tqdm.write(x, file=self.file)

    def flush(self):
        pass


@contextlib.contextmanager
def redirect_stdout(file=None):
    if file is None:
        file = sys.stderr
    old_stdout = file
    sys.stdout = DummyFile(file)
    yield


def print_cfg(cfg):
    print(json.dumps(cfg, indent=4, ensure_ascii=False))


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
    elif path.suffix == ".parquet":
        import pyarrow.parquet as pq
        table = pq.read_table(path)
        df = table.to_pandas()
        return df.to_dict(orient='records')
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
                out[key][value["idx"]] = value
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


import torch
from torch import nn


class QwenModel(nn.Module):
    pretrained_path = "D:/models/Qwen2.5-VL-3B-Instruct"

    def __init__(self, pretrained_path):
        super(QwenModel, self).__init__()
        from transformers import (
            Qwen2_5_VLForConditionalGeneration,
            AutoTokenizer,
            AutoProcessor,
        )

        # default: Load the model on the available device(s)
        if pretrained_path is None:
            pretrained_path = QwenModel.pretrained_path
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            pretrained_path, torch_dtype="auto", device_map="cpu"
        )
        self.processor = AutoProcessor.from_pretrained(pretrained_path)
        from qwen_vl_utils import process_vision_info

        self.processor.process_vision_info = process_vision_info

    def forward(self, question, image):

        messages = []
        for q, i in zip(question, image):
            messages.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": i, "max_pixels": 1920*1080},
                            {"type": "text", "text": q},
                        ],
                    }
                ]
            )

        # Preparation for batch inference
        texts = [
            self.processor.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True
            )
            for msg in messages
        ]
        image_inputs, video_inputs = self.processor.process_vision_info(messages)
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        # Batch Inference
        generated_ids = self.model.generate(
            **inputs, max_new_tokens=128, do_sample=False, temperature=None
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_texts


class OpenClipModel:
    pretrained_path = r"laion2b_s34b_b79k"

    def __init__(self, pretrained_path):
        if pretrained_path is None:
            pretrained_path = OpenClipModel.pretrained_path
        import open_clip

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained=pretrained_path
        )
        self.model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        self.output_tokens = False

    def forward(self, keywards, images):
        if self.output_tokens:
            self.model.visual.output_tokens = True
        keywards = self.tokenizer(keywards).cuda()
        images = [self.preprocess(img) for img in images]
        images = torch.stack(images, dim=0).cuda()
        self.model = self.model.cuda()
        with torch.no_grad(), torch.amp.autocast("cuda"):
            image_features = self.model.encode_image(images)
            if self.output_tokens:
                image_features = image_features[1] @ self.model.visual.proj
            text_features = self.model.encode_text(keywards)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            if self.output_tokens:
                B, N, D = image_features.shape
                image_features = image_features.reshape(B * N, D)
            sim = image_features @ text_features.T
            if self.output_tokens:
                sim = sim.reshape(B, N, -1).max(dim=1).values
            return sim  # (B, M)


class ClipModel:
    def __init__(self, pretrained_path):
        pretrained_path = r"D:\models\ViT-B-32.pt"
        if pretrained_path is None:
            pretrained_path = ClipModel.pretrained_path
        self.model, self.preprocess = clip.load(pretrained_path, device="cuda")

    def forward(self, keywards, images):
        keywards = clip.tokenize(keywards).cuda()
        images = [self.preprocess(img) for img in images]
        images = torch.stack(images, dim=0).cuda()

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = self.model.encode_image(images)
            text_features = self.model.encode_text(keywards)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            sim = image_features @ text_features.T
            return sim  # (B, N, M)


class APIModel:
    def __init__(self, model_name):
        load_dotenv()
        base_url = os.environ.get("OPENAI_BASE_URL", None)
        if base_url is None:
            api_version = "2024-10-21"
            self.client = AsyncAzureOpenAI(api_version=api_version)
        else:
            self.client = AsyncOpenAI()
        self.model_name = model_name

    async def forward(self, user_text, frames=None):
        content = [{"type": "text", "text": user_text}]
        if frames is not None:
            if not isinstance(frames, list):
                frames = [frames]
            for frame in frames:
                img = image2base64(frame)
                content.append({"type": "image_url", "image_url": {"url": img}})
        
        out = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": content}],
            temperature=0,
            frequency_penalty=0,
            presence_penalty=0,
            max_tokens=4096,
            timeout=999,
        )
        out = out.choices[0].message
        return out.content


def create_model(model_type, pretrained_path=None):
    model_type_map = {
        "qwenvl": QwenModel,
        "clip": ClipModel,
        "open_clip": OpenClipModel,
        "api": APIModel,
    }
    if model_type not in model_type_map:
        raise ValueError(f"Unsupported model type: {model_type}")
    return model_type_map[model_type](pretrained_path)


def list2dict(path, level=1):
    """level == 1: qid -> list of answers
    level == 2: qid -> idx -> answer
    """
    list_data = jsonlines.open(path, "r")
    input_data = defaultdict(dict)
    for data in list_data:
        input_data[data["qid"]][data["idx"]] = data["answer"]
    if level == 1:
        for key in input_data:
            input_data[key] = sorted(input_data[key].items(), key=lambda x: x[0])
    return input_data

# def parse_json(pred, list=False):
#     pred = pred.split("```json")[-1].split("```")[0]
#     try:
#         pred = json.loads(pred)
#     except json.JSONDecodeError:
#         try:
#             start = "{" if not list else "["
#             end = "}" if not list else "]"
#             pred = pred.split(start)[1].split(end)[0]
#             pred = start + pred + end
#             pred = json.loads(pred)
#         except Exception as e:
#             print(f"Error parsing JSON: {e}")
#             # return {} if not list else []
#             return None
#     return pred

def remove_json_comments(s):
    # 去除 // 后面的内容
    return re.sub(r'//.*', '', s)

def parse_json(pred, list=False):
    _raw = pred
    pred = pred.split("```json")[-1].split("```")[0]
    try:
        pred = json.loads(pred)
    except json.JSONDecodeError:
        try:
            start = "{" if not list else "["
            end = "}" if not list else "]"
            pred = pred.split(start)[1].split(end)[0]
            pred = start + pred + end
            pred = remove_json_comments(pred)
            # pred = pred.replace("'", '"').replace("‘", '"').replace("’", '"')
            pred = json.loads(pred)
        except Exception as e:
            if list:
                try:
                    pred = pred.replace("[", "").replace("]", "").split(",")
                    pred = [pred.strip().replace("\"", "") for pred in pred]
                    return pred
                except Exception as e2:
                    e = e2
            else:
                try:
                    pred = pred.replace("\n", "").replace(",}", "}")
                    pred = json.loads(pred)
                    return pred
                except Exception as e2:
                    pass
            print(f"Error parsing JSON: {_raw}")
            # return {} if not list else []
            return None
    return pred


def parse_list(list_data):
    out = set()
    for d in list_data:
        if isinstance(d, str):
            if "-" in d:
                start, end = map(int, d.split("-"))
                out.update(list(range(start, end + 1)))
            else:
                out.add(int(d))
        else:
            assert isinstance(d, int), f"Invalid type in list_data: {d}"
            out.add(d)
    return sorted(list(out))


def load_jsonl2dict(path):
    list_data = jsonlines.open(path, "r")
    from collections import defaultdict

    processed = defaultdict(dict)
    for data in list_data:
        if "qid" not in data:
            processed[data["vid"]][data["idx"]] = data
        else:
            processed[data["qid"]][data["idx"]] = data
    return processed


def get_frame(video_path, fps: int, return_idx=False):
    vr = decord.VideoReader(str(video_path))
    # s = frame / origin_fps
    # new_frame = s * fps = frame * fps / origin_fps
    # step = int(frame / new_frame) = int(origin_fps / fps)
    origin_fps = vr.get_avg_fps()
    ratio = origin_fps / fps
    idx = list(range(0, len(vr), int(ratio)))
    if idx[-1] != len(vr) - 1:
        idx.append(len(vr) - 1)
    video = vr.get_batch(idx)
    video = video.cpu().numpy()
    video = [Image.fromarray(v) for v in video]
    if return_idx:
        return video, idx
    return video  # (C, H, W)

def resize_image(img, width=640, height=480):
    w, h = img.size

    scale_w = width / w
    scale_h = height / h
    scale = min(scale_w, scale_h, 1.0)  # 只缩小，不放大

    new_w = int(w * scale)
    new_h = int(h * scale)

    img_resized = img.resize((new_w, new_h), Image.LANCZOS)
    return img_resized
    
def get_frame_by_idx(video_path, idx, fps=1):
    vr = decord.VideoReader(str(video_path))
    origin_fps = vr.get_avg_fps()
    ratio = origin_fps / fps
    origin_idx = list(range(0, len(vr), int(ratio)))
    if origin_idx[-1] != len(vr) - 1:
        origin_idx.append(len(vr) - 1)
    idx = [origin_idx[i] for i in idx]
    video = vr.get_batch(idx)
    video = video.cpu().numpy()
    video = [Image.fromarray(v) for v in video]
    r_height, r_width = smart_resize(video[0].height, video[0].width, 1, 640*480, 1920*1080)
    video = [resize_image(m, r_width, r_height) for m in video] # only valid for videomme-long
    return video  # (C, H, W)

def get_video_size(video_path, fps=1):
    vr = decord.VideoReader(str(video_path))
    origin_fps = vr.get_avg_fps()
    ratio = origin_fps / fps
    origin_idx = list(range(0, len(vr), int(ratio)))
    if origin_idx[-1] != len(vr) - 1:
        origin_idx.append(len(vr) - 1)
    return len(origin_idx)

class LazyFrameLoader:
    @classmethod
    def create(cls, video_path, fps, batch_size=1):
        vr = decord.VideoReader(str(video_path))
        origin_fps = vr.get_avg_fps()
        ratio = origin_fps / fps
        idx = list(range(0, len(vr), int(ratio)))
        if idx[-1] != len(vr) - 1:
            idx.append(len(vr) - 1)
        idx2 = list(range(len(idx)))
        idx = chunk(idx, batch_size)
        idx2 = chunk(idx2, batch_size)
        return [cls(video_path, i, i2) for i, i2 in zip(idx, idx2)]
    
    def __init__(self, path, idx, idx2):
        self._idx = idx
        self.idx = idx2 # 以1fs为单位的索引
        self.path = path
        # self.vr = vr

    def __len__(self):
        return len(self.idx)
    
    def load(self, return_idx=False):
        vr = decord.VideoReader(str(self.path))
        video = vr.get_batch(self._idx)
        video = video.cpu().numpy()
        video = [Image.fromarray(v) for v in video]
        if return_idx:
            return video, self.idx
        return video  # (C, H, W)

def generate_table(row_names, data_dict, filter=True):
    # 获取列名
    data_dict = {key: data_dict[key] for key in sorted(data_dict.keys())}
    column_names = []
    for key in data_dict.keys():
        if filter:
            if all([ans.lower() == "not know" for ans in data_dict[key]["answer"]]):
                continue
        column_names.append(key)

    # 确保每列的数据长度与行名称长度一致
    for column in column_names:
        if len(data_dict[column]["answer"]) != len(row_names):
            raise ValueError(
                f"Data length for column '{column}' does not match the number of row names."
            )

    # 创建Markdown表格的标题行
    header_row = "| |" + " | ".join([str(name) for name in column_names]) + " |"

    # 创建Markdown表格的分隔行
    separator_row = "| " + " | ".join(["---"] * (len(column_names) + 1)) + " |"

    # 创建Markdown表格的内容行
    content_rows = []
    for i, row_name in enumerate(row_names):
        row_data = [data_dict[column]["answer"][i] for column in column_names]
        if filter and all([ans.lower() == "not know" for ans in row_data]):
            continue
        content_row = "| " + " | ".join([row_name] + row_data) + " |"
        content_rows.append(content_row)

    # 将所有行合并为最终的Markdown表格
    markdown_table = "\n".join([header_row, separator_row] + content_rows)

    return markdown_table


image_split = {
    1: [1, 1],
    2: [1, 2],
    3: [1, 3],
    4: [2, 2],
    5: [2, 3],
    6: [2, 3],
    7: [2, 4],
    8: [2, 4],
    9: [3, 3],
    10: [3, 4],
    11: [3, 4],
    12: [3, 4],
    13: [4, 4],
    14: [4, 4],
    15: [4, 4],
    16: [4, 4],
    17: [4, 5],
    18: [4, 5],
    19: [4, 5],
    20: [4, 5],
    21: [5, 5],
    22: [5, 5],
    23: [5, 5],
    24: [5, 5],
    25: [5, 5],
    26: [5, 6],
    27: [5, 6],
    28: [5, 6],
    29: [5, 6],
    30: [5, 6],
    31: [5, 7],
    32: [5, 7],
    33: [5, 7],
    34: [5, 7],
    35: [5, 7],
    36: [6, 6],
    37: [5, 8],
    38: [5, 8],
    39: [5, 8],
    40: [5, 8],
    41: [6, 7],
    42: [6, 7],
    43: [5, 9],
    44: [5, 9],
    45: [5, 9],
    46: [6, 8],
    47: [6, 8],
    48: [6, 8],
    49: [7, 7],
}


# def make_grid(image_list, max_frame=8, pad_width=10):
#     row_num, col_num = best_layout(min(len(image_list), max_frame), *image_list[0].size)
#     nrow = col_num
#     images = [torch.from_numpy(np.array(image)) for image in image_list]
#     if len(images) < nrow * row_num:
#         images += [torch.zeros_like(images[0])] * (nrow * row_num - len(images))
#     else:
#         idx = np.linspace(0, len(images) - 1, nrow * row_num).astype(int)
#         images = [images[i] for i in idx]
#     images = torch.stack(images, dim=0).permute(0, 3, 1, 2)
#     out = tv_make_grid(images, nrow=nrow, padding=pad_width)
#     out = out.permute(1, 2, 0).numpy().astype(np.uint8)
#     out = Image.fromarray(out)
#     return out

def make_grid(image_list, max_frame=8, pad_width=10):
    assert len(image_list) <= max_frame, "Image list length exceeds max_frame"
    assert max_frame <= 49, "max_frame should be less than 49"
    if len(image_list) > max_frame:
        idx = np.linspace(0, len(image_list) - 1, max_frame).astype(int)
        idx = sorted(set(idx))
        image_list = [image_list[i] for i in idx]
    row_num, col_num = image_split[len(image_list)]
    # nrow = len(image_list) // row_num + (1 if len(image_list) % row_num != 0 else 0)
    nrow = col_num
    images = [torch.from_numpy(np.array(image)) for image in image_list]
    images += [torch.zeros_like(images[0])] * (nrow * row_num - len(image_list))
    images = torch.stack(images, dim=0).permute(0, 3, 1, 2)
    out = tv_make_grid(images, nrow=nrow, padding=pad_width)
    out = out.permute(1, 2, 0).numpy().astype(np.uint8)
    out = Image.fromarray(out)
    return out


def make_anno_grid(image_list, boxes, max_frame=8, pad_width=10):
    frame_limit = max(list(image_split.keys()))
    assert max_frame <= frame_limit, f"max_frame should be less than {frame_limit}"
    if len(image_list) > max_frame:
        idx = np.linspace(0, len(image_list) - 1, max_frame).astype(int)
        idx = sorted(set(idx))
        image_list = [image_list[i] for i in idx]
        boxes = [boxes[i] for i in idx]
    row_num, col_num = image_split[len(image_list)]
    origin_w, origin_h = image_list[0].size
    total_w = int(origin_w * col_num + pad_width * (col_num - 1))
    total_h = int(origin_h * row_num + pad_width * (row_num - 1))
    out = Image.new("RGB", (total_w, total_h), (0, 0, 0))
    for i, (img, box) in enumerate(zip(image_list, boxes)):
        crop = img.copy()
        draw = ImageDraw.Draw(crop)
        for b in box:
            draw.rectangle(b, outline="red", width=2)
        row_idx = i // col_num
        col_idx = i % col_num
        x1 = col_idx * (origin_w + pad_width) + pad_width
        y1 = row_idx * (origin_h + pad_width) + pad_width
        out.paste(crop, (x1, y1))
    return out


def make_crop_grid(image_list, boxes, max_frame=8, pad_width=10):
    frame_limit = max(list(image_split.keys()))
    assert max_frame <= frame_limit, f"max_frame should be less than {frame_limit}"
    if len(image_list) > max_frame:
        idx = np.linspace(0, len(image_list) - 1, max_frame).astype(int)
        idx = sorted(set(idx))
        image_list = [image_list[i] for i in idx]
        boxes = [boxes[i] for i in idx]
    row_num, col_num = image_split[len(image_list)]
    # nrow = len(image_list) // row_num + (1 if len(image_list) % row_num != 0 else 0)
    nrow = col_num
    # images = [torch.from_numpy(np.array(image)) for image in image_list]
    # images += [torch.zeros_like(images[0])] * (nrow * row_num - len(image_list))
    image_w, image_h = image_list[0].size
    xyxy_list = [
        get_xyxy(box) if len(box) != 0 else (0, 0, image_w, image_h) for box in boxes
    ]
    ratio = [math.fabs((xyxy[0] - xyxy[2]) / (xyxy[1] - xyxy[3])) for xyxy in xyxy_list]
    avg_ratio = np.mean(ratio)
    origin_w = int(image_w * avg_ratio)
    origin_h = int(image_h)
    total_w = int(origin_w * col_num + pad_width * (col_num - 1))
    total_h = int(origin_h * row_num + pad_width * (row_num - 1))
    out = Image.new("RGB", (total_w, total_h), (0, 0, 0))
    for i, (img, xyxy) in enumerate(zip(image_list, xyxy_list)):
        crop = crop_img(img, xyxy, (origin_w, origin_h))
        row_idx = i // col_num
        col_idx = i % col_num
        x1 = col_idx * (origin_w + pad_width) + pad_width
        y1 = row_idx * (origin_h + pad_width) + pad_width
        out.paste(crop, (x1, y1))
    # images = torch.stack(images, dim=0).permute(0, 3, 1, 2)
    # out = tv_make_grid(images, nrow=nrow, padding=pad_width)
    # out = out.permute(1, 2, 0).numpy().astype(np.uint8)
    # out = Image.fromarray(out)
    return out


def adjust_crop_box_to_aspect_strict(x1, y1, x2, y2, origin_w, origin_h):
    crop_w = x2 - x1
    crop_h = y2 - y1
    origin_ratio = origin_w / origin_h
    crop_ratio = crop_w / crop_h

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    if crop_ratio < origin_ratio:
        # 需要增宽
        new_w = int(origin_ratio * crop_h)
        diff = new_w - crop_w
        new_x1 = x1 - diff // 2
        new_x2 = x2 + diff - diff // 2

        # 边界调整
        if new_x1 < 0:
            new_x2 += -new_x1
            new_x1 = 0
        if new_x2 > origin_w:
            shift = new_x2 - origin_w
            new_x1 -= shift
            new_x2 = origin_w
        # 再次保证不超界
        new_x1 = max(0, new_x1)
        new_x2 = min(origin_w, new_x2)
        new_y1, new_y2 = y1, y2

    elif crop_ratio > origin_ratio:
        # 需要增高
        new_h = int(crop_w / origin_ratio)
        diff = new_h - crop_h
        new_y1 = y1 - diff // 2
        new_y2 = y2 + diff - diff // 2

        # 边界调整
        if new_y1 < 0:
            new_y2 += -new_y1
            new_y1 = 0
        if new_y2 > origin_h:
            shift = new_y2 - origin_h
            new_y1 -= shift
            new_y2 = origin_h
        # 再次保证不超界
        new_y1 = max(0, new_y1)
        new_y2 = min(origin_h, new_y2)
        new_x1, new_x2 = x1, x2

    else:
        # 比例一致
        new_x1, new_y1, new_x2, new_y2 = x1, y1, x2, y2

    # 最终取整
    return (
        int(round(new_x1)),
        int(round(new_y1)),
        int(round(new_x2)),
        int(round(new_y2)),
    )


def get_xyxy(boxes):
    x1, y1, x2, y2 = float("inf"), float("inf"), float("-inf"), float("-inf")
    for box in boxes:
        x1 = min(x1, box[0], box[2])
        y1 = min(y1, box[1], box[3])
        x2 = max(x2, box[0], box[2])
        y2 = max(y2, box[1], box[3])
    return x1, y1, x2, y2


def crop_img(img, boxes_or_xyxy, origin_wh=None):
    """
    crop image to the bounding box or xyxy coordinates, and resize to the original size.
    Args:
        img (PIL.Image): The input image to crop.
        boxes_or_xyxy (list or tuple): The bounding box coordinates in the format
            [(x1, y1, x2, y2), ...] or (x1, y1, x2, y2).
        origin_wh (tuple): The original width and height of the image (optional).
    """
    if len(boxes_or_xyxy) == 0:
        return img
    if isinstance(boxes_or_xyxy[0], list) or isinstance(boxes_or_xyxy[0], tuple):
        x1, y1, x2, y2 = get_xyxy(boxes_or_xyxy)
    else:
        x1, y1, x2, y2 = boxes_or_xyxy
    if origin_wh is not None:
        origin_w, origin_h = origin_wh
    else:
        origin_w, origin_h = img.size

    new_x1, new_y1, new_x2, new_y2 = adjust_crop_box_to_aspect_strict(
        x1, y1, x2, y2, origin_w, origin_h
    )

    crop = img.crop((new_x1, new_y1, new_x2, new_y2))
    resized = crop.resize((origin_w, origin_h), Image.LANCZOS)
    return resized


def annote_frame_idx(image, frame_idx):
    """
    在PIL图片左上角标注帧的序号

    :param image: PIL.Image.Image对象
    :param frame_idx: 当前帧序号（从1开始）
    :return: 标注后的PIL图片
    """
    # 复制图片，避免修改原图
    img = image.copy()
    draw = ImageDraw.Draw(img)
    text = f"Frame {frame_idx}"
    font = ImageFont.truetype("arial.ttf", 40)  # 字体大小为20
    # 计算文本大小
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    # 位置
    x, y = 10, 10
    # 画底色矩形（可选，增强可读性）
    margin = 10
    # draw.rectangle([x - margin, y - margin, x + text_width + margin, y + text_height + margin], fill=(255, 255, 255, 128))
    # 画文字
    draw.text(
        (x, y),
        text,
        font=font,
        fill=(0, 0, 0),
        stroke_width=2,
        stroke_fill=(255, 255, 255),
    )
    return img


def annote_box(image, box):
    crop = image.copy()
    draw = ImageDraw.Draw(crop)
    for b in box:
        draw.rectangle(b, outline="red", width=2)
    return crop


def send_post_request(json_file):
    url = "https://validation-server.onrender.com/api/upload/"
    headers = {"Content-Type": "application/json"}
    with open(json_file, "r") as f:
        data = json.load(f)
    response = requests.post(url, headers=headers, json=data)
    return response


import math


def best_layout(n, w, h):
    best = None
    for r in range(1, n + 1):
        c = n // r
        if c == 0:
            continue
        used = r * c
        rem = n - used
        ratio = (c * w) / (r * h)
        diff = abs(ratio - 1)
        # 优先rem最小，其次diff最小
        key = rem + diff
        if (best is None) or (key < best[0]):
            best = (key, r, c)
    _, r, c = best
    return r, c
