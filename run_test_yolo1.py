from pathlib import Path
from runner import Runner
from ultralytics import YOLO
import decord
from task_utils import get_frame

decord.bridge.set_bridge("torch")
# video_path = "D:/datasets/nextqa/NExTVideo"

target_fps = 1


def detect_video(runner, **data):
    _video_path = Path(runner.dataset.config.video_path).joinpath(data["video_path"])
    frame = get_frame(_video_path, target_fps)
    results = model(frame, stream=True, verbose=False, batch=256)
    ret = []
    for result in results:
        sub_ret = {}
        boxes = result.boxes
        for c in boxes.cls.unique():
            n = (boxes.cls == c).sum()  # detections per class
            sub_ret[result.names[int(c)]] = int(n)
        ret.append(sub_ret)
    return {"vid": data["vid"], "detect": ret}


if __name__ == "__main__":
    # exp_name = "0404"
    exp_name = "0522"
    # dataset_name = "nextmc_test"
    dataset_name = "egoschema_subset"
    
    output_path = f"./outputs/{exp_name}/yolo.jsonl"
    model = YOLO("yolo11n.pt").to("cuda:0")
    runner = Runner(detect_video, output_path, iter_key="vid", dataset=dataset_name)
    runner()
