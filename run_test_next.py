from model.builder import build_model
from PIL import Image
import torch
import copy
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
)
from llava.mm_utils import (
    tokenizer_image_token,
)
from llava.conversation import conv_templates
# out = load_llava_video()
model, tokenizer, processor = build_model("D:/models/LLaVA-Video-7B-Qwen2", "video_llava", delay_load=False)
image = r"D:\work\实时对话\VideoTree-e2e\test\9072405003\frame_1.jpg"
# text = "Are there men, children, or women in white in the picture? Please answer separately."
# text = "Is the lady making a hand gesture like a thumbs up?"
# text = "Please provide a brief answer to the following questions: 1: Is the lady holding a dog in her arms?  \n2: Does the lady have a club in her hand?  \n3: Is the lady seen walking towards or away from the camera?  \n4: Is there any visible cosmetic item being used by the lady?  \n5: Is the lady making a hand gesture like a thumbs up?, output the answer directly without other text, if there is not enough information in the frame, just answer 'not know', seperate an with line breaks. Output Example:\n1: answer1\n2: answer2\n3: answer3"
text = "Is there a mix of yellow and blue on the cotton stick?"
conv_template = "qwen_1_5"
image = Image.open(image) 
video = processor.preprocess(image, return_tensors="pt")[
    "pixel_values"
]
video = video.to(torch.bfloat16)
question = DEFAULT_IMAGE_TOKEN + text
conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()
input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX)
model = model.to("cuda")
input_ids = torch.as_tensor(input_ids).unsqueeze(0).cuda()
out = model.generate(
    inputs=input_ids,
    images=video,
    modalities=["image"],
    max_new_tokens=10000,
)
print(tokenizer.decode(out[0], skip_special_tokens=True))
# {"man": ["man finish taking the boy's photo", "man smile"], "boy": ["boy gets his photo taken"], "lady in white": ["lady in white receives photo"]}