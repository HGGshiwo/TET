from runner import Runner
import numpy as np
from utils import load_data
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from task_utils import get_frame
from pathlib import Path
from utils import chunk

def tensor_to_dict(result):
    out = {}
    for key in result.keys():
        if isinstance(result[key], torch.Tensor):
            out[key] = result[key].cpu().numpy().tolist()
        else:
            out[key] = result[key]
    return out

def frame_select(runner, **data):
    pred_obj = detect_data[data["qid"]]["pred"]
    # if "question" in pred_obj:
        # pred_obj = pred_obj["question"] # only use object apear in question
    pred_obj = list(set([item for item_list in pred_obj.values() for item in item_list]))
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
        chunk_size = max(32, len(pred_obj)) // len(pred_obj)
        chunk_images = chunk(images, chunk_size)
        for i, image in enumerate(chunk_images):
            image_inputs = [img.copy() for img in image for _ in pred_obj] # image: [1,2,3] -> [1,1,1,2,2,2,3,3,3]
            text_inputs = [f"{obj}." for obj in pred_obj] * len(image) # text: [1,2,3] -> [1,2,3,1,2,3,1,2,3]
            model_inputs = processor(images=image_inputs, text=text_inputs, return_tensors="pt", padding=True).to(DEVICE)
            with torch.no_grad():
                outputs = grounding_model(**model_inputs)
                outputs2 = processor.post_process_grounded_object_detection(
                    outputs,
                    model_inputs.input_ids,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                    target_sizes=[img.size[::-1] for img in image_inputs]
                )
            for i, output in enumerate(outputs2):
                if len(output["scores"]) == 0:
                    continue
                frame_idx = i // len(pred_obj)
                obj_idx = pred_obj[i % len(pred_obj)]
                if results.get(frame_idx) is None:
                    results[frame_idx] = {}
                if results[frame_idx].get(obj_idx) is None:
                    results[frame_idx][obj_idx] = []
                results[frame_idx][obj_idx].append(tensor_to_dict(output))
            
    return {
        "qid": data["qid"],
        "results": results,
        "last": len(images)
    }


if __name__ == "__main__":
    # exp_name = "0601"
    # exp_name = "0607"
    # exp_name = "0609"
    # exp_name = "0614"
    # exp_name = "0619"
    exp_name = "0621"
    
    # dataset_name = "egoschema_subset"
    dataset_name = "nextmc_test"
    
    param_type = "low" # 空缺默认是low
    # param_type = "high"
    
    model_type = "tiny" # 空缺默认是tiny
    # model_type = "base"
    
    output_path = f"./outputs/{exp_name}/dino_out_{dataset_name}_{param_type}_{model_type}.jsonl"
    
    # detect_data = load_data("./outputs/0601/dino_gpt-4.1_nextmc_test.jsonl")
    detect_data = load_data("./outputs/0604/dino_gpt-4.1_nextmc_test_option2.jsonl")
    # detect_data = load_data("./outputs/0607/dino_gpt-4.1_egoschema_subset_option2.jsonl")
    
    if param_type == "low": 
        box_threshold=0.4
        text_threshold=0.3
    elif param_type == "high":
        box_threshold=0.6
        text_threshold=0.5
    
    GROUNDING_MODEL = f"D:/models/grounding-dino-{model_type}"
    # SAM2_CHECKPOINT = "D:/models/sam2.1_hiera_large.pt"
    # SAM2_MODEL_CONFIG = "D:/work/实时对话/VideoTree-e2e2/Grounded-SAM-2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
    DEVICE = "cuda"

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
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(DEVICE)

    runner = Runner(frame_select, output_path, iter_key="qid", dataset=dataset_name)
    runner()
