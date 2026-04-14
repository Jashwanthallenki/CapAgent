import os
import concurrent.futures
from openai import OpenAI
from tqdm import tqdm
import PIL.Image
import gradio_client


# ------------------ LLMChatClient with fallback ------------------
class LLMChatClient:

    def __init__(self, api_key=None, models=None):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key or os.environ.get("OPENROUTER_API_KEY")
        )
        # List of free fallback models (you can extend this)
        self.models = models or [
            "deepseek/deepseek-chat-v3-0324:free",
            "qwen/qwen2.5-7b-instruct:free",
            "nousresearch/nous-capybara-7b:free",
            "mistralai/mistral-7b-instruct:free"
        ]

    def _try_models(self, func, *args, **kwargs):
        """Try all models in fallback order until success."""
        last_error = None
        for model in self.models:
            try:
                print(f"🔄 Trying model: {model}")
                return func(model, *args, **kwargs)
            except Exception as e:
                print(f"⚠️ Model {model} failed: {e}")
                last_error = e
        raise RuntimeError(f"All models failed. Last error: {last_error}")

    def text_completion(self, prompt, temperature=0, max_tokens=512):
        def _call(model, prompt, temperature, max_tokens):
            resp = self.client.completions.create(
                model=model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].text

        return self._try_models(_call, prompt, temperature, max_tokens)

    def chat_completion(self, messages, temperature=0, max_tokens=1024):
        def _call(model, messages, temperature, max_tokens):
            resp = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content

        return self._try_models(_call, messages, temperature, max_tokens)

    def handle_text_completion(self, request):
        return {"id": request['id'], "result": self.text_completion(request['prompt'])}

    def handle_chat_completion(self, request):
        return {"id": request['id'], "result": self.chat_completion(request['messages'])}

    def process_requests_multithreaded(self, requests, max_parallel_requests=8):
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel_requests) as executor:
            futures = []
            with tqdm(total=len(requests), desc="Processing requests") as pbar:
                for request in requests:
                    if request['type'] == 'text':
                        futures.append(executor.submit(self.handle_text_completion, request))
                    elif request['type'] == 'chat':
                        futures.append(executor.submit(self.handle_chat_completion, request))

                for future in concurrent.futures.as_completed(futures):
                    results.append(future.result())
                    pbar.update(1)
        return results


# ------------------ MLLMChatClient with fallback ------------------
class MLLMChatClient:

    def __init__(self, api_key=None, models=None):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key or os.environ.get("OPENROUTER_API_KEY")
        )
        self.models = models or [
            "deepseek/deepseek-chat-v3-0324:free",
            "qwen/qwen2.5-7b-instruct:free",
            "nousresearch/nous-capybara-7b:free",
            "mistralai/mistral-7b-instruct:free"
        ]

    def _try_models(self, func, *args, **kwargs):
        last_error = None
        for model in self.models:
            try:
                print(f"🔄 Trying model: {model}")
                return func(model, *args, **kwargs)
            except Exception as e:
                print(f"⚠️ Model {model} failed: {e}")
                last_error = e
        raise RuntimeError(f"All multimodal models failed. Last error: {last_error}")

    def chat_completion(self, messages, temperature=0, max_tokens=512, timeout=None):
        def _call(model, messages, temperature, max_tokens, timeout):
            resp = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout  # ✅ pass timeout
            )
            return resp.choices[0].message.content

        return self._try_models(_call, messages, temperature, max_tokens, timeout)





    

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

    
