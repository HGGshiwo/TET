from task_utils import create_model, get_frame
import decord
import json
import torch
from PIL import Image
from run_test_clip2 import SortSimilarity
vid = "3083302557"
map = json.load(open("D:/datasets/nextqa/map_vid_vidorID.json", "r"))
video_path = f"D:/datasets/nextqa/NExTVideo/{map[vid]}.mp4"

video = get_frame(video_path, 1)
# video = [Image.open(r"D:\work\实时对话\VideoTree-e2e2\CLIP.png")]
# model = create_model("clip")
# model.output_tokens = True
# sim = model.forward(["woman"], video)
# # sim = model.forward(["a diagram", "a dog", "a cat"], video)
# print((sim[0] * 100).softmax(dim=-1))
# sim = sim.max(dim=-1).values
# sim = torch.topk(sim, int(0.5 * sim.shape[0]), dim=0).indices
# sim = list(sim.cpu().numpy())
# print(sim)

model = create_model("clip2")
sim = model.forward(["woman"], video)
out = SortSimilarity(sim, ["woman"], 0.5 * sim.shape[0])
print(sorted(list(out.keys())))