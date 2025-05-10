from ultralytics import YOLO
from pathlib import Path
from PIL import Image
from run_test_yolo1 import get_frame
import decord

decord.bridge.set_bridge("torch")

# Load an official or custom model
# model = YOLO("yolo11n.pt")  # Load an official Detect model
model = YOLO("yolo11n-seg.pt")  # Load an official Segment model
# model = YOLO("yolo11n-pose.pt")  # Load an official Pose model
# model = YOLO("path/to/best.pt")  # Load a custom trained model

# Perform tracking with the model
# path = "D:/work/实时对话/VideoTree-e2e/test/4188739935"
# image = [Image.open(img) for img in Path(path).glob("*.jpg")]
# image = "D:/datasets/nextqa/NExTVideo/1201/4188739935.mp4"
image = "D:/datasets/nextqa/NExTVideo/1103/3557601110.mp4"
target_fps = 10
frame = get_frame(image, target_fps)
model = model.to("cuda:0")  
results = model.track(frame, show=True, stream=True) # Tracking with default tracker
for r in results:
    print(r.boxes)  # Print box coordinates
    print(r.masks)  # Print mask coordinates
    print(r.keypoints)  # Print keypoint coordinates
# print(results)
# results = model.track("https://youtu.be/LNwODJXcvt4", show=True, tracker="bytetrack.yaml")  # with ByteTrack