import PIL
import gradio as gr
import tempfile
import os
import requests
import io
import base64

from gradio_toggle import Toggle
from capagent.instruction_augmenter import InstructionAugmenter
from capagent.tools import count_words
from run import run_agent

#IMGUR_CLIENT_ID = "YOUR_IMGUR_CLIENT_ID"

instruction_augmenter = InstructionAugmenter()

EXAMPLES = [
    # example 1
    [
        "Create a detailed description of the image, focusing on the central figure seated on an ornate throne.",
        "assets/figs/charles_on_the_throne.png"
    ],
    # example 2
    [
        "Captioning this image no more than 10 words.", 
        "assets/figs/cat.png"
    ],

    # example 3
    [
        "Captioning this image in a funny tone.", 
        "assets/figs/funny_cat.png"
    ],

    # example 4
    [
        "Captioning this image with a sad tone and no more than three sentences.", 
        "assets/figs/sad_person.png"
    ],

    # example 5
    [
        "Captioning this news photo.", 
        "assets/figs/trump_assassination.png"
    ],
    
    # example 6
    [
        f"Please describe the image within 30 words.", 
        "assets/figs/statue_of_liberty.png"
    ],

    # example 7
    [
        f"Please describe this cab.", 
        "assets/figs/cybercab.png"
    ],

    # example 8
    [
        "Please describe this image.", 
        "assets/figs/venom.png"
    ],

    # example 9
    [
        "Please describe the spatial relationship in this image.", 
        "assets/figs/living_room.png"
    ]
]


'''def upload_to_imgbb(image_path):
    """
    Upload an image to ImgBB and return the public URL.
    Requires an API key from https://api.imgbb.com/.
    """
    IMGBB_API_KEY = os.getenv("IMGBB_API_KEY")
    if not IMGBB_API_KEY:
        raise Exception("IMGBB_API_KEY not set in environment variables.")

    with open(image_path, "rb") as file:
        payload = {
            "key": IMGBB_API_KEY,
            "image": base64.b64encode(file.read()),
        }
    response = requests.post("https://api.imgbb.com/1/upload", payload)

    try:
        data = response.json()
    except Exception:
        raise Exception(f"Invalid JSON from ImgBB: {response.text}")

    if data.get("status") == 200 and "url" in data["data"]:
        return data["data"]["url"]
    else:
        raise Exception(f"ImgBB upload failed: {data}")'''


'''def upload_to_imgbb(image):
    """
    Upload a PIL image to ImgBB and return the public URL.
    Requires an API key from https://api.imgbb.com/.
    """
    IMGBB_API_KEY = os.getenv("IMGBB_API_KEY")
    if not IMGBB_API_KEY:
        raise Exception("IMGBB_API_KEY not set in environment variables.")

    # Save the PIL image to memory instead of disk
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    payload = {
        "key": IMGBB_API_KEY,
        "image": base64.b64encode(buffer.read()),
    }

    response = requests.post("https://api.imgbb.com/1/upload", payload)

    try:
        data = response.json()
        print(data)
    except Exception:
        raise Exception(f"Invalid JSON from ImgBB: {response.text}")

    if data.get("status") == 200 and "url" in data["data"]:
        return data["data"]["url"]
    else:
        raise Exception(f"ImgBB upload failed: {data}")'''

'''def upload_to_imgbb(image):
    """
    Upload a PIL image. 
    1. Try ImgBB first.
    2. If ImgBB fails, fall back to FreeImageHost.
    """

    # Save image to memory
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    # Encode base64 once for reuse
    encoded_image = base64.b64encode(buffer.read())

    # --- 1. Try ImgBB ---
    try:
        imgbb_key = os.getenv("IMGBB_API_KEY")
        if not imgbb_key:
            raise Exception("IMGBB_API_KEY not set")

        payload = {"key": imgbb_key, "image": encoded_image}
        resp = requests.post("https://api.imgbb.com/1/upload", payload, timeout=15)
        data = resp.json()

        if data.get("status") == 200 and "url" in data["data"]:
            return data["data"]["url"]
        else:
            raise Exception(f"ImgBB upload failed: {data}")

    except Exception as e:
        print(f"[WARN] ImgBB failed, falling back. Error: {e}")

    # --- 2. Fallback: FreeImageHost ---
    try:
        freeimg_key = os.getenv("FREEIMAGE_API_KEY")
        if not freeimg_key:
            raise Exception("FREEIMAGE_API_KEY not set")

        payload = {"key": freeimg_key, "action": "upload", "source": encoded_image}
        resp = requests.post("https://freeimage.host/api/1/upload", data=payload, timeout=15)
        data = resp.json()

        if data.get("status_code") == 200 and "image" in data:
            return data["image"]["url"]
        else:
            raise Exception(f"FreeImageHost upload failed: {data}")

    except Exception as e2:
        raise Exception(f"Both ImgBB and fallback failed: {e2}")
'''

def upload_to_imgbb(image):
    """
    Upload a PIL image:
    1. Try ImgBB first.
    2. Fall back to GitHub (Ananya-Bijja/capagent) if ImgBB fails.
    """
    github_path="images/img.png"
    # Save image to memory
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    encoded_image = base64.b64encode(buffer.read())

    # --- 1. ImgBB ---
    try:
        imgbb_key = os.getenv("IMGBB_API_KEY")
        if not imgbb_key:
            raise Exception("IMGBB_API_KEY not set")

        payload = {"key": imgbb_key, "image": encoded_image}
        resp = requests.post("https://api.imgbb.com/1/upload", data=payload, timeout=15)
        data = resp.json()

        if data.get("status") == 200 and "url" in data["data"]:
            return data["data"]["url"]
        else:
            raise Exception(f"ImgBB failed: {data}")

    except Exception as e:
        print(f"[WARN] ImgBB failed: {e}")

    # --- 2. GitHub ---
    try:
        github_repo = "Ananya-Bijja/capagent"  # Repo is now internal
        token = os.getenv("GITHUB_TOKEN")
        if not token:
            raise Exception("GITHUB_TOKEN not set")

        buffer.seek(0)
        content = base64.b64encode(buffer.read()).decode()
        url = f"https://api.github.com/repos/{github_repo}/contents/{github_path}"
        data = {"message": "Add image", "content": content}
        headers = {"Authorization": f"token {token}"}

        resp_check = requests.get(url,json=data, headers=headers)
        sha = resp_check.json()["sha"] if resp_check.status_code == 200 else None

        data = {"message": "Add/update image", "content": content}
        if sha:
            data["sha"] = sharesp_check = requests.get(url, headers=headers)
        sha = resp_check.json()["sha"] if resp_check.status_code == 200 else None

        data = {"message": "Add/update image", "content": content}
        if sha:
            data["sha"] = sha
        resp = requests.put(url, json=data, headers=headers)
        resp.raise_for_status()
        return resp.json()["content"]["download_url"]

    except Exception as e:
        raise Exception(f"All uploads failed: {e}")

def generate_complex_instruction(query: str, image: PIL.Image.Image, is_search: bool):
    try:
        print(image)
        public_url=upload_to_imgbb(image)
        a=instruction_augmenter.generate_complex_instruction(image,public_url, query, is_search=is_search, timeout=20)
        #print(a)
        return a
    except Exception as e:
        return e


'''def process_query(query: str, image: PIL.Image.Image) -> str:
    try:
        # Create temporary directory for image processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded images to temp directory if any
            
            image_path = os.path.join(temp_dir, f"image.png")
            image.save(image_path)
            image_paths = [image_path]
            
            result, messages = run_agent(
                user_query=query, 
                working_dir=temp_dir, 
                image_paths=image_paths
            )

            return result, messages
        
    except Exception as e:
        return f"Error occurred: {str(e)}", []'''

def process_query(query: str,image: PIL.Image.Image) -> str:
    try:
        '''# Save image persistently
        save_path = os.path.join("uploaded_images", "image.png")
        os.makedirs("uploaded_images", exist_ok=True)
        image.save(save_path)
        print("DEBUG: Received image path:", image)
        print("DEBUG: Type:", type(image))

        # Upload to public host
        public_url = upload_to_imgbb(save_path)
        print("Public image URL:", public_url)
'''
        print("entered process query")

        public_url = upload_to_imgbb(image)
        result, messages = run_agent(
            user_query=query, 
            working_dir="uploaded_images", 
            image_paths=[public_url]  # pass public URL
        )

        return result, messages
        
    except Exception as e:
        return f"Error occurred: {str(e)}", []


def launch_gradio_demo():
    # Create the Gradio interface

    with gr.Blocks() as demo:
        gr.Markdown("<h1><a href='https://github.com/xin-ran-w/CapAgent'>CapAgent</a></h1>")
        gr.Markdown("CapAgent is a tool-using agent for image captioning. It can generate professional instructions for image captioning, and use tools to generate more accurate captions.")
        gr.Markdown("## Usage")
        gr.Markdown("1. Enter your simple instruction and upload image to interact with the CapAgent.")
        gr.Markdown("2. Click the button 'Generate Professional Instruction' to generate a professional instruction based on your instruction.")
        gr.Markdown("3. Click the button 'Send' to generate a caption for the image based on your professional instruction.")
        with gr.Row():
            
            with gr.Column():
                image_input = gr.Image(height=256, image_mode="RGB", type="pil", label="Image")
                query_input = gr.Textbox(label="User Instruction", placeholder="e.g., 'Captioning an image with more accurate event information'", lines=2, submit_btn="Send")
                
                with gr.Blocks():
                    pro_instruction_input = gr.Textbox(label="Professional Instruction", submit_btn="Send")

                web_search_toggle = Toggle(
                    label="Use Google Search and Google Lens",
                    value=False,
                    color="green",
                    interactive=True,
                )
            
                with gr.Row():
                    complex_button = gr.Button("Generate Professional Instruction")
                    clear_button = gr.Button("Clear")

                gr.Examples(
                    examples=EXAMPLES,
                    inputs=[query_input, image_input],
                )

            with gr.Column():
                output_textbox = gr.Textbox(label="Agent Response", lines=10)
                cot_textbox = gr.Chatbot(label="Chain of Thought Messages", type='messages', min_height=600)

        gr.Markdown("## Contact")
        gr.Markdown("If you have any questions or suggestions, please contact me at <a href='mailto:wangxr@bupt.edu.cn'>wangxr@bupt.edu.cn</a>.")

        complex_button.click(
            generate_complex_instruction, 
            inputs=[query_input, image_input, web_search_toggle], 
            outputs=pro_instruction_input
        )

        pro_instruction_input.submit(
            process_query, 
            inputs=[pro_instruction_input, image_input], 
            outputs=[output_textbox, cot_textbox]
        )

        query_input.submit(
            process_query, 
            inputs=[query_input, image_input], 
            outputs=[output_textbox, cot_textbox]
        )

        clear_button.click(lambda: [None, None, None, None, None], outputs=[output_textbox, cot_textbox, pro_instruction_input, image_input, query_input])

        output_textbox.change(
            lambda x: gr.update(label=f"Agent Response {count_words(x)} words" if x else "Agent Response"), 
            inputs=output_textbox, 
            outputs=output_textbox
        )
    
    
    # Launch the demo
    demo.launch(
        share=True,                    # Create a public link
        #server_name="10.112.104.168",       # Make available on all network interfaces
        #server_port=7861,                  # Default Gradio port,
        server_name="0.0.0.0",  # Use local host
        server_port=7862,
        debug=True
    )

if __name__ == "__main__":
    launch_gradio_demo()