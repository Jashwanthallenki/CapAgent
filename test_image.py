from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import requests

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

img_url = "https://github.com/Ananya-Bijja/capagent/blob/main/images/img.png"
image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")

inputs = processor(image, return_tensors="pt")
out = model.generate(**inputs)
caption = processor.decode(out[0], skip_special_tokens=True)
print("Caption:", caption)
