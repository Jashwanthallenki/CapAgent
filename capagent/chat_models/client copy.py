import os
import concurrent.futures
import openai
import PIL.Image
import gradio_client
from openai import OpenAI

from tqdm import tqdm
from io import BytesIO


from capagent.utils import encode_pil_to_base64


class LLMChatClient:

    '''def __init__(self, url=None, api_key='EMPTY', using_gpt4o=False) -> None:
        if not using_gpt4o:
            self.client = openai.Client(base_url=url, api_key=api_key)
            self.model = 'default'
        else:
            print("Using GPT-4o as LLM Chat Client")
            self.client = openai.Client(api_key=os.environ['OPENAI_API_KEY'])
            self.model = 'gpt-4o'''
    
    '''def __init__(self, url=None, api_key=None, model="deepseek/deepseek-chat-v3-0324:free"):
        # Use OpenRouter API
        openai.api_base = "https://openrouter.ai/api/v1"
        openai.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.model = model'''
    
    def __init__(self, url=None, api_key=None, model="deepseek/deepseek-chat-v3-0324:free"):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key or os.environ["OPENROUTER_API_KEY"]
        )
        self.model = model
    
    def text_completion(self, prompt, temperature=0, max_tokens=512):
        
        response = self.client.completions.create(
            model=self.model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
            
        return response.choices[0].text

    def chat_completion(self, messages):

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
            max_tokens=1024,
        )

        return response.choices[0].message.content


    def handle_text_completion(self, request):
        prompt = request['prompt']
        result = self.text_completion(prompt)
        return {"id": request['id'], "result": result}

    def handle_chat_completion(self, request):
        messages = request['messages']
        result = self.chat_completion(messages)
        return {"id": request['id'], "result": result}

    def process_requests_multithreaded(self, requests, max_parallel_requests=8, show_progress=False):
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel_requests) as executor:
            futures = []
            # Add progress bar with total number of requests
            with tqdm(total=len(requests), desc="Processing requests") as pbar:
                for request in requests:
                    if request['type'] == 'text':
                        futures.append(executor.submit(self.handle_text_completion, request))
                    elif request['type'] == 'chat':
                        futures.append(executor.submit(self.handle_chat_completion, request))

                # As requests complete, update the progress bar
                for future in concurrent.futures.as_completed(futures):
                    results.append(future.result())
                    pbar.update(1)  # Update progress bar after each request completion
        return results
    

class MLLMChatClient:
    
    '''def __init__(self, url=None, api_key="EMPTY", using_gpt4o=False) -> None:
        if not using_gpt4o:
            self.client = openai.Client(base_url=url, api_key=api_key)
            self.model = 'default'
        else:
            self.client = openai.Client(api_key=os.environ['OPENAI_API_KEY'])
            self.model = 'gpt-4o'''

    '''def __init__(self, api_key=None, model="deepseek/deepseek-chat-v3-0324:free"):
        openai.api_base = "https://openrouter.ai/api/v1"
        openai.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.model = model'''
    
    
    def __init__(self, api_key=None, model="deepseek/deepseek-chat-v3-0324:free"):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key or os.environ.get("OPENROUTER_API_KEY")
        )
        self.model = model


    def is_url(self, image):
        return image.startswith("http")

    def chat_completion(self, messages, temperature=0, max_tokens=512, timeout=20):

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout
        )
                    
        return response.choices[0].message.content


    

class SegmentationClient:

    def __init__(self, url) -> None:
        self.client = gradio_client.Client(url)

    def segment_region(self, image, points):
        return self.client.predict(image, points)

    

# llm_client = LLMChatClient(url='http://10.112.8.137:30000/v1')
#llm_client = LLMChatClient(using_gpt4o=True)
llm_client = LLMChatClient()

try:
    # mllm_client = MLLMChatClient(url='http://10.112.8.137:31000/v1')
    #mllm_client = MLLMChatClient(using_gpt4o=True)
    mllm_client=MLLMChatClient()
except Exception as e:
    raise Warning("MLLMChatClient is not initialized.")
    mllm_client = None

def test_single_image_chat_completion():
    image = PIL.Image.open("assets/image.png").convert("RGB")
    print(mllm_client.single_image_chat_completion("What is this image?", image))

def test_multithreaded_requests():
    requests = [
        {"id": 1, "type": "text", "prompt": "The capital of France is"},
        {"id": 2, "type": "chat", "messages": [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "What's your name?"}
        ]},
    ]   

    results = llm_client.process_requests_multithreaded(requests)

    for result in results:
        print(result)


if __name__ == "__main__":

    # To launch the client
    # python -m sglang.launch_server --model-path $HUGGINGFACE_DIR/Meta-Llama-3.1-8B-Instruct

    test_single_image_chat_completion()
    test_multithreaded_requests()

    
