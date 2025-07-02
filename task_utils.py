import decord
# from model.builder import build_model
import supervision as sv
from PIL import Image, ImageDraw
decord.bridge.set_bridge("torch")
import torch
from openai import OpenAI, AsyncOpenAI
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
)
from llava.mm_utils import (
    tokenizer_image_token,
)
from llava.conversation import conv_templates
import copy
from dotenv import load_dotenv
from utils import image2base64
from collections import defaultdict
import jsonlines
import json
from PIL import Image
from torchvision.utils import make_grid as tv_make_grid
import numpy as np
import torch.nn.functional as F
import clip
from LVNet.src.open_clip import create_model_and_transforms
from transformers import AutoProcessor, Blip2ForImageTextRetrieval, AddedToken
# from lavis.models import load_model_and_preprocess
# from lavis.processors import load_processor
import cv2
import math
from PIL import Image, ImageDraw, ImageFont

class LLavaModel:
    def __init__(self, pretrained_path):
        if pretrained_path is None:
               pretrained_path = "D:/models/LLaVA-Video-7B-Qwen2"
        self.model, self.tokenizer, self.processor = build_model(
            pretrained_path, "video_llava", delay_load=False
        )
    
    def forward(self):
        video = self.processor.preprocess(video, return_tensors="pt")["pixel_values"]
        video = video.to(torch.bfloat16)
        question = DEFAULT_IMAGE_TOKEN + question
        conv_template = "qwen_1_5"
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX)
        input_ids = torch.as_tensor(input_ids).unsqueeze(0).cuda()
        model = model.cuda()
        modalities = ["image"] if len(video) == 1 else ["video"]
        out = model.generate(
            inputs=input_ids,
            images=video,
            modalities=modalities,
            max_new_tokens=1000,
        )
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

class QwenModel:
    def __init__(self, pretrained_path):
        from transformers import (
            Qwen2_5_VLForConditionalGeneration,
            AutoTokenizer,
            AutoProcessor,
        )
        # default: Load the model on the available device(s)
        if pretrained_path is None:
            pretrained_path = "D:/models/Qwen2.5-VL-3B-Instruct"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            pretrained_path, torch_dtype="auto", device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained("D:/models/Qwen2.5-VL-3B-Instruct")
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
                            {"type": "image", "image": i},
                            {"type": "text", "text": q},
                        ],
                    }
                ]
            )

        # Preparation for batch inference
        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages
        ]
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Batch Inference
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
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
    def __init__(self, pretrained_path):
        if pretrained_path is None:
            pretrained_path = r"laion2b_s34b_b79k"
        import open_clip
        self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained=pretrained_path)
        self.model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
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
            return sim  #(B, M)

class ClipModel:
    def __init__(self, pretrained_path):
        if pretrained_path is None:
            pretrained_path = r"D:\models\ViT-B-32.pt"
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
            return sim  #(B, N, M)  
        
class ClipModel2:
    def __init__(self, pretrained_path=None):
        if pretrained_path is None:
            pretrained_path = r"D:\models\clippy_5k.pt"
        self.clippy, preprocess_train, self.preprocess_val = create_model_and_transforms(
            "clippy-B-16",
            device="cuda",
            pretrained=pretrained_path
        )
        self.clip_size = (224, 224)
    
    def forward(self, keywords, images):
        images = [self.preprocess_val(img) for img in images]
        images = torch.stack(images, dim=0).cuda()
        img_embed = self.clippy.encode_image(images.cuda(), pool=False)[:, 1:]
        
        keyword_embed = self.clippy.text.encode(keywords, convert_to_tensor=True)

        nframe, nimgtokens, channels = img_embed.shape
        keyword_embed = keyword_embed.unsqueeze(1)
        img_embed = img_embed.flatten(0, 1).unsqueeze(0) 

        simmat = F.cosine_similarity(keyword_embed, img_embed, dim=-1).to(torch.float)
        return simmat.reshape(nframe, nimgtokens, -1)  # (B, N, M)

class BlipModel:
    def __init__(self, pretrained_path=None):
        if pretrained_path is None:
            pretrained_path = "D:/models/blip2-itm-vit-g"

        self.model, self.vis_processors, self.text_processors = load_model_and_preprocess("blip2_image_text_matching", "coco", device="cuda", is_eval=True)


    def forward(self, question, image):
        img = torch.stack([self.vis_processors["eval"](m) for m in image], dim=0).to("cuda")
        txt = [self.text_processors["eval"](q) for q in question]

        # using "itm" or image-text matching (for predicting whether match or not)
        itm_output = self.model({"image": img, "text_input": txt}, match_head="itc")
        # itm_scores = torch.nn.functional.softmax(itm_output, dim=1)
        # print(f'The image and text are matched with a probability of {itm_scores[:, 1].item():.3%}')
        return itm_output
    
        # # using "itc" or image-text contrastive (for computing cosine similarity)
        # itc_score = model({"image": img, "text_input": txt}, match_head='itc')
        # print('The image feature and text feature has a cosine similarity of %.4f'%itc_score)


class APIModel:
    def __init__(self, model_name):
        load_dotenv()
        self.client = AsyncOpenAI()
        self.model_name = model_name
    
    async def forward(self, user_text, frame=None):
        content = [{"type": "text", "text": user_text}]
        if frame is not None:
            img = image2base64(frame)
            content.append({"type": "image_url", "image_url": {"url": img}})
        out = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": content}],
            temperature=0,
            frequency_penalty=0,
            presence_penalty=0,
            max_tokens=4096,
            timeout=999
        )
        out = out.choices[0].message
        return out.content

def create_model(model_type, pretrained_path=None):
    model_type_map = {
        "llava": LLavaModel,
        "qwenvl": QwenModel,
        "clip": ClipModel,
        "clip2": ClipModel2,
        "blip": BlipModel,
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


def parse_json(pred, list=False):
    pred = pred.split("```json")[-1].split("```")[0]
    try:
        pred = json.loads(pred)
    except json.JSONDecodeError:
        try:
            start = "{" if not list else "["
            end = "}" if not list else "]"
            pred = pred.split(start)[1].split(end)[0]
            pred = start + pred + end
            pred = json.loads(pred)
        except Exception as e:
            print(f"Error parsing JSON: {e}")
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

import decord
decord.bridge.set_bridge("torch")

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

def make_grid(image_list, max_frame=8, pad_width=10):
    assert max_frame <= 49, "max_frame should be less than 49"
    if len(image_list) > max_frame:
        idx = np.linspace(0, len(image_list) - 1, max_frame).astype(int)
        idx = list(set(idx))
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
        idx = list(set(idx))
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
        idx = list(set(idx))
        image_list = [image_list[i] for i in idx]
        boxes = [boxes[i] for i in idx]
    row_num, col_num = image_split[len(image_list)]
    # nrow = len(image_list) // row_num + (1 if len(image_list) % row_num != 0 else 0)
    nrow = col_num
    # images = [torch.from_numpy(np.array(image)) for image in image_list]
    # images += [torch.zeros_like(images[0])] * (nrow * row_num - len(image_list))
    image_w, image_h = image_list[0].size
    xyxy_list = [get_xyxy(box) if len(box) != 0 else (0, 0, image_w, image_h) for box in boxes]
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
    return int(round(new_x1)), int(round(new_y1)), int(round(new_x2)), int(round(new_y2))

def get_xyxy(boxes):
    x1, y1, x2, y2 = float('inf'), float('inf'), float('-inf'), float('-inf')
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
    text = f'Frame {frame_idx}'
    font = ImageFont.truetype('arial.ttf', 40)  # 字体大小为20
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
    draw.text((x, y), text, font=font, fill=(0, 0, 0), stroke_width=2, stroke_fill=(255, 255, 255))
    return img
    