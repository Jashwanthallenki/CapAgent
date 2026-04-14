import argparse
from functools import partial
import cv2
import requests
import os
from io import BytesIO
from PIL import Image
import numpy as np
from pathlib import Path


import warnings

import torch

# prepare the environment
# os.system("python setup.py build develop --user")
# os.system("pip install packaging==21.3")
# os.system("pip install gradio==3.50.2")


warnings.filterwarnings("ignore")

import gradio as gr

from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from groundingdino.util.inference import annotate, load_image, predict
import groundingdino.datasets.transforms as T

from huggingface_hub import hf_hub_download



# Use this command for evaluate the Grounding DINO model


def load_model_hf(model_config_path, weights_path, device='cpu'):
    args = SLConfig.fromfile(model_config_path) 
    model = build_model(args)
    args.device = device

    # cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(weights_path, map_location='cpu')
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(weights_path, log))
    _ = model.eval()
    return model    

def image_transform_grounding(init_image):
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image, _ = transform(init_image, None) # 3, h, w
    return init_image, image

def image_transform_grounding_for_vis(init_image):
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
    ])
    image, _ = transform(init_image, None) # 3, h, w
    return image


model = None

def detection(input_image, grounding_caption, box_threshold, text_threshold):
    if isinstance(input_image, str):
        input_image = Image.open(input_image)

    init_image = input_image.convert("RGB")
    original_size = init_image.size

    _, image_tensor = image_transform_grounding(init_image)
    image_pil: Image = image_transform_grounding_for_vis(init_image)

    # run grounidng
    boxes, logits, phrases = predict(model, image_tensor, grounding_caption, box_threshold, text_threshold, device='cpu')
    
    annotated_frame = annotate(image_source=np.asarray(image_pil), boxes=boxes, logits=logits, phrases=phrases)
    image_with_box = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
    
    boxes = boxes.cpu().numpy().tolist()
    logits = logits.cpu().numpy().tolist()
    
    return image_with_box, {
        "bboxes": boxes,
        "logits": logits,
        "phrases": phrases
    }

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounding DINO demo", add_help=True)
    parser.add_argument("--debug", action="store_true", help="using debug mode")
    parser.add_argument("--share", action="store_true", help="share the app")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="host")
    parser.add_argument("--port", type=int, default=8080, help="port")
    parser.add_argument("--config", type=str, default="/mnt/sdc/huggingface/model_hub/GroundingDINO/GroundingDINO_SwinT_OGC.cfg.py", help="config file")
    parser.add_argument("--ckpt", type=str, default="/mnt/sdc/huggingface/model_hub/GroundingDINO/groundingdino_swint_ogc.pth", help="checkpoint file")
    args = parser.parse_args()

    model = load_model_hf(args.config, args.ckpt)

    demo = gr.Interface(fn=detection, 
                        inputs=[
                            gr.Image(type="filepath"),
                            "text",
                            gr.Number(value=0.35),
                            gr.Number(value=0.25)
                        ],
                        outputs=[gr.Image(type="pil"), "json"]
                    )
    
    demo.launch(share=True, server_name=args.host, server_port=args.port, show_error=True)

