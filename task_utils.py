import decord
# from model.builder import build_model

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
from lavis.models import load_model_and_preprocess
from lavis.processors import load_processor
    
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


def make_grid(image_list, max_frame=8, pad_width = 10):
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
    }
    assert max_frame < 17, "max_frame should be less than 17"
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