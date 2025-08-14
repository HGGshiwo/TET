from utils import create_model
model = create_model("qwenvl")
from PIL import Image
PROMPT = """
From the following list: peas, water, salt, ingredient, tool, knife, fork, measuring cup, pan, spoon, plate, bowl, select only the items that are present in the image. Output your answer as a list. If none are present, output an empty list.
"""
image = Image.open(r"D:\work\实时对话\VideoTree-e2e2\outputs\dino_vis\004a7f7e-9e83-431f-bc98-859cf9024e93\10.png")
out = model.forward([PROMPT], [image])
print(out)