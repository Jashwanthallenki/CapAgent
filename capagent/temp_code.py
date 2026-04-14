from PIL import Image

import subprocess
import sys
from PIL import Image

# Check and install required packages
try:
    import transformers
except ImportError:
    # Install transformers and torch with pip
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet', 'transformers', 'torch'])
    import transformers

# Import necessary libraries after installation
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

# Load model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# Generate caption with focus prompt
prompt = "a central figure seated on an ornate throne"
inputs = processor(image_1, text=prompt, return_tensors="pt").to(device)
out = model.generate(**inputs, max_new_tokens=100)
caption = processor.decode(out[0], skip_special_tokens=True)

# Output the detailed caption
print(caption)
