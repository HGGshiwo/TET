GPU_IDS = [1, 2]
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, GPU_IDS))
from runner import MultiGPURunner
from utils import load_data
import torch
from utils import create_model, parse_json
from utils import load_data, save_data, print_cfg


PROMPT = """
From the following list: [list], select only the items that are present in the image. Output your answer as a list. If none are present, output an empty list.
"""


def frame_select(runner, model, data):
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
            return [{"qid": data["qid"], "idx": idx, "out": []} for idx in data["frame"].idx]

        batch_prompt = [PROMPT.replace("[list]", str(pred_obj))] * len(data["frame"].idx)
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
    # box_threshold = cfg["box_threshold"]
    # text_threshold = cfg["text_threshold"]
    # single_obj = cfg["single_obj"]  # 是否只使用单个对象
    # GROUNDING_MODEL = cfg["grounding_model"]
    model_type = cfg.get("model_type", "qwenvl")
    model_path = cfg["model_path"]
    exp_name = cfg.get("exp_name", None)

    obj_cfg = load_data(cfg["obj"])
    dataset_name = obj_cfg["dataset_name"]
    if exp_name is None:
        exp_name = obj_cfg["exp_name"]
        cfg["exp_name"] = exp_name

    save_data(cfg, f"./outputs/{exp_name}/dino.yml")
    print_cfg(cfg)
    output_path = f"./outputs/{exp_name}/dino.jsonl"
    detect_data = load_data(f"./outputs/{obj_cfg['exp_name']}/obj.jsonl")

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    runner = MultiGPURunner(
        frame_select,
        output_path,
        iter_key="qid",
        iter_frame=True,
        dataset=dataset_name,
        video_fps=1,
        batch_size=8,
    )
    runner.detect_data = detect_data
    runner.question_only = question_only
    from functools import partial
    model_class = partial(create_model, model_type=model_type, pretrained_path=model_path)
    runner(model_class=model_class, gpu_ids=list(range(len(GPU_IDS))))
