from runner import Runner
import numpy as np
from utils import load_data
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from task_utils import get_frame
from pathlib import Path
from utils import chunk, load_data, save_data


def tensor_to_dict(result):
    out = {}
    for key in result.keys():
        if isinstance(result[key], torch.Tensor):
            out[key] = result[key].cpu().numpy().tolist()
        else:
            out[key] = result[key]
    return out


def frame_select(runner, **data):
    if data["qid"] not in detect_data:
        pred_obj = []
    else:
        pred_obj = detect_data[data["qid"]]["pred"]
        # if "question" in pred_obj:
        # pred_obj = pred_obj["question"] # only use object apear in question
        if question_only:
            pred_obj = pred_obj["question"]
        else:
            pred_obj = list(
                set([item for item_list in pred_obj.values() for item in item_list])
            )
    # pred_obj = pred_obj["question"]
    pred_obj = [obj.lower() for obj in pred_obj]
    # filter for egoschema subset
    pred_obj = [obj for obj in pred_obj if obj.lower() != "c"]
    # setup the input image and text prompt for SAM 2 and Grounding DINO
    # VERY important: text queries need to be lowercased + end with a dot
    # text = [f"{obj}." for obj in pred_obj]
    pred_obj = [obj.lower() for obj in pred_obj]

    video_path = Path(runner.dataset.config.video_path).joinpath(data["video_path"])
    images = get_frame(video_path, 1)

    if len(pred_obj) == 0:
        results = {}
    else:
        results = {}
        if single_obj:
            chunk_size = max(32, len(pred_obj)) // len(pred_obj)
        else:
            chunk_size = 32
        chunk_images = chunk(images, chunk_size)
        outs = []
        for i, image in enumerate(chunk_images):
            if single_obj:
                # image: [1,2,3] -> [1,1,1,2,2,2,3,3,3]
                image_inputs = [img.copy() for img in image for _ in pred_obj]
                # text: [1,2,3] -> [1,2,3,1,2,3,1,2,3]
                text_inputs = [f"{obj}." for obj in pred_obj] * len(image)
            else:
                # image: [1,2,3] -> [1,2,3]
                image_inputs = [img.copy() for img in image]
                text_inputs = [" ".join([f"{obj}." for obj in pred_obj]) for _ in image]
            model_inputs = processor(
                images=image_inputs, text=text_inputs, return_tensors="pt", padding=True
            ).to(DEVICE)
            with torch.no_grad():
                outputs = grounding_model(**model_inputs)
                out = processor.post_process_grounded_object_detection(
                    outputs,
                    model_inputs.input_ids,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                    target_sizes=[img.size[::-1] for img in image_inputs],
                )
                outs.extend(out)
        for j, output in enumerate(outs):
            if len(output["scores"]) == 0:
                continue
            if single_obj:
                frame_idx = j // len(pred_obj)
                obj_idx = pred_obj[j % len(pred_obj)]
                if results.get(frame_idx) is None:
                    results[frame_idx] = {}
                if results[frame_idx].get(obj_idx) is None:
                    results[frame_idx][obj_idx] = []
                results[frame_idx][obj_idx].append(tensor_to_dict(output))
            else:
                frame_idx = j
                results[frame_idx] = tensor_to_dict(output)
                
    return {"qid": data["qid"], "results": results, "last": len(images)}


if __name__ == "__main__":
    cfg = load_data("./configs/dino.yml")
    question_only = cfg["question_only"]  # 是否只使用问题中的对象
    box_threshold = cfg["box_threshold"]
    text_threshold = cfg["text_threshold"]
    single_obj = cfg["single_obj"]  # 是否只使用单个对象
    GROUNDING_MODEL = cfg["grounding_model"]
    exp_name = cfg.get("exp_name", None)
    
    obj_cfg = load_data(cfg["obj"])
    dataset_name = obj_cfg["dataset_name"]
    if exp_name is None:
        exp_name = obj_cfg["exp_name"]
        cfg["exp_name"] = exp_name
    
    save_data(cfg, f"./outputs/{exp_name}/dino.yml")
    output_path = f"./outputs/{exp_name}/dino.jsonl"
    detect_data = load_data(f"./outputs/{obj_cfg['exp_name']}/obj.jsonl")

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # environment settings
    # use bfloat16
    torch.autocast(device_type=DEVICE, dtype=torch.bfloat16).__enter__()

    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # build SAM2 image predictor
    # sam2_checkpoint = SAM2_CHECKPOINT
    # model_cfg = SAM2_MODEL_CONFIG
    # sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE)
    # sam2_predictor = SAM2ImagePredictor(sam2_model)

    # build grounding dino from huggingface
    model_id = GROUNDING_MODEL
    processor = AutoProcessor.from_pretrained(model_id)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(
        DEVICE
    )

    runner = Runner(frame_select, output_path, iter_key="qid", dataset=dataset_name)
    runner()
