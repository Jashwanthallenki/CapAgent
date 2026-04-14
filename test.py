import os
import requests
import base64
from serpapi import GoogleSearch

IMGBB_API_KEY = "e4be84891d196209d609902e3eae43a5"  # Get from https://api.imgbb.com/

def upload_to_imgbb(image_path):
    url = "https://api.imgbb.com/1/upload"

    with open(image_path, "rb") as file:
        encoded_image = base64.b64encode(file.read())

    payload = {
        "key": IMGBB_API_KEY,
        "image": encoded_image
    }

    res = requests.post(url, data=payload)
    data = res.json()

    if "data" in data:
        return data["data"]["url"]
    else:
        raise Exception(f"ImgBB upload failed: {data}")

def search_with_google_lens(image_url):
    api_key = os.getenv("SERPAPI_KEY")  # Make sure this is set in your environment
    params = {
        "engine": "google_lens",
        "url": image_url,
        "api_key": api_key
    }
    search = GoogleSearch(params)
    return search.get_dict()

if __name__ == "__main__":
    image_path = "test.jpg"  # Replace with your image
    public_url = upload_to_imgbb(image_path)
    print("✅ Uploaded to ImgBB:", public_url)

    results = search_with_google_lens(public_url)
    print("🔍 Google Lens results:", results)
