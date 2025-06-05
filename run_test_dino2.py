from runner import Runner
import numpy as np
from utils import load_data
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from task_utils import get_frame
from pathlib import Path
from utils import chunk

rate = []
total, valid = 0, 0

def tensor_to_list(result):
    out = {}
    for key in result.keys():
        if isinstance(result[key], torch.Tensor):
            out[key] = result[key].cpu().numpy().tolist()
        else:
            out[key] = result[key]
    return out

def frame_select(runner, **data):
    pred_obj = detect_data[data["qid"]]["pred"]
    if "question" in pred_obj:
        pred_obj = pred_obj["question"] # only use object apear in question
    pred_obj = [obj.lower() for obj in pred_obj]
    # setup the input image and text prompt for SAM 2 and Grounding DINO
    # VERY important: text queries need to be lowercased + end with a dot
    text = [f"{obj}." for obj in pred_obj]
    
    video_path = Path(runner.dataset.config.video_path).joinpath(data["video_path"])
    images = get_frame(video_path, 1)
    
    if len(text) == 0:
        results = {}
    else:
        results = {}
        inputs = chunk(images, 8)
        outs = []
        for chunck_images in inputs:
            model_inputs = processor(images=chunck_images, text=[" ".join(text) for _ in chunck_images], return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                outputs = grounding_model(**model_inputs)
                out = processor.post_process_grounded_object_detection(
                    outputs,
                    model_inputs.input_ids,
                    # box_threshold=0.6,
                    # text_threshold=0.5,
                    box_threshold=0.4,
                    text_threshold=0.3,
                    target_sizes=[image.size[::-1] for image in chunck_images]
                )
                outs.extend(out)
        frame_obj_map = np.zeros((len(outs), len(text)), dtype=bool)
        for i, result in enumerate(outs):
            if len(result["labels"]) == 0:
                continue # no objects detected, skip this image
            if not skip_object:
                if len(result["labels"]) == len(pred_obj):
                    results[i] = tensor_to_list(result)
                continue
            # record objects that appear in the image
            for label in result["labels"]:
                try:
                    idx = pred_obj.index(label.lower())
                except ValueError:
                    idx = -10
                    for j, obj in enumerate(pred_obj):
                        if label.lower() in obj:
                            idx = j
                            break
                if idx == -1:
                    continue
                frame_obj_map[i, idx] = True
        if skip_object:
            # skip objects that not apear in all images
            legal_obj = np.any(frame_obj_map, axis=0)
            frame_obj_map = frame_obj_map[:, legal_obj]
            relevant_idx = np.all(frame_obj_map, axis=1)
            relevant_idx = np.nonzero(relevant_idx)[0].tolist()
            for i in relevant_idx:
                results[i] = tensor_to_list(outs[i])
            
    relevant_idx = list(results.keys())   
    if len(relevant_idx) == 0:
        relevant_idx = sorted(list(set(np.linspace(0, len(images)-1, 8).astype(int).tolist())))
    else:
        global valid
        valid += 1
        rate.append(len(relevant_idx) / len(images))
        
    global total
    total += 1
    
    return {
        "qid": data["qid"],
        "relevant_idx": relevant_idx,
        "last": len(images),
        "results": results
    }


if __name__ == "__main__":
    # exp_name = "0601"
    exp_name = "0604"
    
    skip_object = True # skip objects that not apear in all images
    # skip_object = False
    
    # dataset_name = "egoschema_subset"
    dataset_name = "nextmc_test"
    
    end = "_skip" if skip_object else ""
    output_path = f"./outputs/{exp_name}/dino_select{end}.jsonl"
    
    # detect_data = load_data("./outputs/0601/dino_gpt-4.1_nextmc_test.jsonl")
    detect_data = load_data("./outputs/0604/dino_gpt-4.1_nextmc_test_option2.jsonl")
    
    GROUNDING_MODEL = "D:/models/grounding-dino-tiny"
    SAM2_CHECKPOINT = "D:/models/sam2.1_hiera_large.pt"
    SAM2_MODEL_CONFIG = "D:/work/实时对话/VideoTree-e2e2/Grounded-SAM-2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
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
    print(np.mean(rate))
    print(f"valid: {valid/total}({valid}/{total})")
