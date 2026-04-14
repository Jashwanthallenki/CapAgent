import base64
import json
from io import BytesIO


def encode_pil_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def save_jsonlines(data, file_path):
    with open(file_path, 'w') as f:
        for item in data:
            json.dump(item, f, indent=4)
            f.write('\n')