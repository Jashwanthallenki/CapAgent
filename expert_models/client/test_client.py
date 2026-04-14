from gradio_client import Client, file
from PIL import Image
import tempfile
import requests #new
import shutil
import webbrowser

def test_detection_client():
    client = Client("http://0.0.0.0:8081")

    image = Image.open("fried_chicken.png")
    with tempfile.NamedTemporaryFile(delete=True, suffix=".jpg") as tmp_file:
        image.save(tmp_file.name, 'JPEG')
        image = tmp_file.name
        result_image_file, result_json = client.predict(file(image), "food", 0.3, 0.3)
        print(result_image_file)
        print(result_json)


'''def test_depth_client():
    client = Client("http://127.0.0.1:7860")
    image = Image.open("fried_chicken.png")
    with tempfile.NamedTemporaryFile(delete=True, suffix=".jpg") as tmp_file:
        image.save(tmp_file.name, 'JPEG')
        image = tmp_file.name
        _, grayscale_depth_map, _ = client.predict(file(image), api_name="/on_submit")
        print(grayscale_depth_map)'''

def test_depth_client():
    client = Client("http://127.0.0.1:7860")
    image = Image.open("fried_chicken.png")

    import os
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        temp_path = tmp_file.name

    image.save(temp_path, 'JPEG')
    try:
        _, grayscale_depth_map, _ = client.predict(file(temp_path), api_name="/on_submit")
        print(grayscale_depth_map)
        output_path = os.path.join(os.getcwd(), "saved_output.png")
        shutil.copy(grayscale_depth_map, output_path)
        print(f"Saved to: {output_path}")

        # Open the saved image (optional)
        webbrowser.open(f"file://{output_path}")

    finally:
        os.remove(temp_path)


    

if __name__ == "__main__":
    test_depth_client()

    
