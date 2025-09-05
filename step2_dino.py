import os

GPU_IDS = [0]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(gpu) for gpu in GPU_IDS])
from runner import MultiGPURunner
from utils import load_data
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from utils import get_frame
from pathlib import Path
from utils import chunk, load_data, save_data, print_cfg, create_model, parse_json
import copy
from functools import partial

PROMPT = """
From the following list: [list], select only the items that are present in the image. Output your answer as a list. If none are present, output an empty list.
"""


def tensor_to_dict(result):
    out = {}
    for key in result.keys():
        if isinstance(result[key], torch.Tensor):
            out[key] = result[key].cpu().numpy().tolist()
        else:
            out[key] = result[key]
    return out


def expand_text(model_inputs, num):
    # text: [1,2,3] -> [1,2,3,1,2,3,1,2,3]
    for key in ["input_ids", "attention_mask", "token_type_ids"]:
        dims = [num] + [1] * (len(model_inputs[key].shape) - 1)
        model_inputs[key] = model_inputs[key].repeat(*dims)
    return model_inputs


def expand_image(model_inputs, num):
    # image: [1,2,3] -> [1,1,1,2,2,2,3,3,3]
    for key in ["pixel_values", "pixel_mask"]:
        model_inputs[key] = model_inputs[key].repeat_interleave(num, dim=0)
    return model_inputs


def frame_select_dino(runner, model, **data):
    try:
        grounding_model = model
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

        if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        if data["qid"] not in runner.detect_data:
            pred_obj = []
        else:
            pred_obj = runner.detect_data[data["qid"]]["pred"]
            if runner.question_only:
                pred_obj = pred_obj["question"]
            else:
                pred_obj = [
                    item for item_list in pred_obj.values() for item in item_list
                ]
                pred_obj = list(set(pred_obj))

        pred_obj = [obj.lower() for obj in pred_obj]
        pred_obj = [obj for obj in pred_obj if obj.lower() != "c"]
        pred_obj = [obj.lower() for obj in pred_obj]

        video_path = Path(runner.dataset.config.video_path).joinpath(data["video_path"])
        images = get_frame(video_path, 1)

        def model_forward(model_inputs):
            # environment settings
            # use bfloat16
            with torch.no_grad():
                try:
                    model_inputs.to(grounding_model.device)
                    outputs = grounding_model(**model_inputs)
                    out = cur_processor.post_process_grounded_object_detection(
                        outputs,
                        model_inputs.input_ids,
                        box_threshold=runner.box_threshold,
                        text_threshold=runner.text_threshold,
                        # target_sizes=[img.size[::-1] for img in image_inputs],
                    )
                    return out
                except Exception as e:
                    print(f"Error processing chunk: {e}")
                    return None

        results = {}
        if len(pred_obj) != 0:
            outs = []
            cur_processor = copy.deepcopy(runner.processor)
            pre_process = partial(cur_processor, padding=True, return_tensors="pt")
            batch_size = 24
            if runner.single_obj:
                pred_obj = pred_obj[:batch_size]
                chunk_size = batch_size // len(pred_obj)
                chunk_images = chunk(images, chunk_size)
                text_cache = [f"{obj}." for obj in pred_obj]
                text_cache = pre_process(images=None, text=text_cache)
                for image in chunk_images:
                    model_inputs = pre_process(images=image, text=None)
                    model_inputs = expand_image(model_inputs, len(pred_obj))
                    text_inputs = expand_text(text_cache.copy(), len(image))
                    model_inputs.update(text_inputs)
                    out = model_forward(model_inputs)
                    outs.extend(out)
                for j, output in enumerate(outs):
                    if len(output["scores"]) == 0:
                        continue
                    frame_idx = j // len(pred_obj)
                    obj_idx = pred_obj[j % len(pred_obj)]
                    if results.get(frame_idx) is None:
                        results[frame_idx] = {}
                    if results[frame_idx].get(obj_idx) is None:
                        results[frame_idx][obj_idx] = []
                    results[frame_idx][obj_idx].append(tensor_to_dict(output))
            else:
                chunk_size = batch_size
                chunk_images = chunk(images, chunk_size)
                text_cahce = [" ".join([f"{obj}." for obj in pred_obj])]
                text_cahce = pre_process(images=None, text=text_cahce)
                for i, image in enumerate(chunk_images):
                    image_inputs = [img.copy() for img in image]
                    model_inputs = pre_process(images=image_inputs, text=None)
                    text_inputs = expand_text(text_cahce.copy(), len(image))
                    model_inputs.update(text_inputs)
                    out = model_forward(model_inputs)
                    outs.extend(out)
                for j, output in enumerate(outs):
                    if len(output["scores"]) == 0:
                        continue
                    frame_idx = j
                    results[frame_idx] = tensor_to_dict(output)

        return {"qid": data["qid"], "results": results, "last": len(images)}
    except Exception as e:
        import traceback
        print(f"Error in {data['qid']}: {e}")
        traceback.print_exc()
        return None


def frame_select_qwen(runner, model, data):
    try:
        if data["qid"] not in runner.detect_data:
            pred_obj = []
        else:
            pred_obj = runner.detect_data[data["qid"]]["pred"]
            if runner.question_only:
                pred_obj = pred_obj["question"]
            else:
                pred_obj = [
                    item for item_list in pred_obj.values() for item in item_list
                ]
                pred_obj = list(set(pred_obj))

        pred_obj = [obj.lower() for obj in pred_obj]
        pred_obj = [obj for obj in pred_obj if obj.lower() != "c"]
        pred_obj = [obj.lower() for obj in pred_obj]

        if len(pred_obj) == 0:
            return [
                {"qid": data["qid"], "idx": idx, "out": []} for idx in data["frame"].idx
            ]

        batch_prompt = [PROMPT.replace("[list]", str(pred_obj))] * len(
            data["frame"].idx
        )
        batch_frame = data["frame"].load()
        model_out = model.forward(batch_prompt, batch_frame)
        ret = []
        for out, idx in zip(model_out, data["frame"].idx):
            try:
                o = parse_json(out, list=True)
            except Exception as e:
                o = None
            ret.append({"qid": data["qid"], "idx": idx, "out": o})
        return ret
    except Exception as e:
        import traceback

        print(f"Error in {data['qid']}: {e}")
        traceback.print_exc()
        return [None for _ in data["frame"].idx]


if __name__ == "__main__":
    cfg = load_data("./configs/dino.yml")
    question_only = cfg["question_only"]  # 是否只使用问题中的对象
    exp_name = cfg.get("exp_name", None)
    model_path = cfg["model_path"]
    obj_cfg = load_data(cfg["obj"])
    dataset_name = obj_cfg["dataset_name"]
    if exp_name is None:
        exp_name = obj_cfg["exp_name"]
        cfg["exp_name"] = exp_name

    save_data(cfg, f"./outputs/{exp_name}/dino.yml")
    print_cfg(cfg)
    output_path = f"./outputs/{exp_name}/dino.jsonl"
    detect_data = load_data(f"./outputs/{obj_cfg['exp_name']}/obj.jsonl")

    kwargs = {
        "output_path": output_path,
        "dataset": dataset_name,
        "video_fps": 1,
        "iter_key": "qid",
        "question_only": question_only,
        "detect_data": detect_data,
    }
    if "qwen" in cfg["model_path"].lower():
        model_type = cfg.get("model_type", "qwenvl")
        new_kwargs = {
            "task": frame_select_qwen,
            "iter_frame": True,
            "batch_size": 8,
            "model_class": partial(
                create_model, model_type=model_type, pretrained_path=model_path
            ),
        }
        kwargs.update(new_kwargs)
        model_class = partial(
            create_model, model_type=model_type, pretrained_path=model_path
        )
    else:
        GROUNDING_MODEL = cfg["model_path"]
        box_threshold = cfg["box_threshold"]
        text_threshold = cfg["text_threshold"]
        processor = AutoProcessor.from_pretrained(GROUNDING_MODEL)
        new_kwargs = {
            "task": frame_select_dino,
            "iter_frame": False,
            "processor": processor,
            "box_threshold": box_threshold,
            "text_threshold": text_threshold,
            "single_obj": cfg["single_obj"],
        }
        kwargs.update(new_kwargs)
        model_class = partial(
            AutoModelForZeroShotObjectDetection.from_pretrained, GROUNDING_MODEL
        )

    runner = MultiGPURunner(**kwargs)
    runner(model_class=model_class, gpu_ids=list(range(len(GPU_IDS))))
