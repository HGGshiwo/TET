import decord
from PIL import Image
from pathlib import Path
import json
decord.bridge.set_bridge("torch")
from utils import chunk
import torch

vid = "3812429626"
map = json.load(open("D:/datasets/nextqa/map_vid_vidorID.json", "r"))
_video_path = f"D:/datasets/nextqa/NExTVideo/{map[vid]}.mp4"
vr = decord.VideoReader(str(_video_path))
fps = vr.get_avg_fps()
frame_idx = list(range(0, len(vr), int(fps * 1)))
if frame_idx[-1] != len(vr) - 1:
    frame_idx.append(len(vr) - 1)
video = vr.get_batch(frame_idx)
path = f"test/{vid}"
Path(path).mkdir(parents=True, exist_ok=True)
# video = chunk(video, 1)
for idx, frame in enumerate(video):
    # frame = frame.float().mean(dim=0).to(torch.uint8)
    img = Image.fromarray(frame.cpu().numpy())
    img.save(f"{path}/frame_{idx}.bmp")