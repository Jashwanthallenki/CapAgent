import re
import os
from autogen.agentchat import ConversableAgent

from capagent.agent import (
    CapAgent, 
    checks_terminate_message
)

from capagent.prompt import (
    ReActPrompt, 
    ASSISTANT_SYSTEM_MESSAGE
)
from capagent.execution import CodeExecutor
from capagent.parse import Parser
from capagent.chat_models.client import mllm_client
from capagent.utils import encode_pil_to_base64


def extract_tool_comments(file_path):
    tool_comments = {}
    with open(file_path, 'r') as file:
        content = file.read()
        
        # Regex to match tool definitions and capture comments (modify based on your comment format)
        matches = re.findall(r'(#.*?\n)?\s*def\s+(\w+)\(', content)
        
        for comment, tool_name in matches:
            tool_comments[tool_name] = comment.strip("# ").strip() if comment else "No comment"
    
    return tool_comments



def run_agent(user_query: str, working_dir: str, image_paths: list[str] = None):
    
    prompt_generator = ReActPrompt()
    
    executor = CodeExecutor(working_dir=working_dir, use_tools=True)
    
    parser = Parser()
    
    if image_paths is not None:
        image_loading_result = executor.loading_images(image_paths)
        if image_loading_result[0] != 0:
            raise Exception(f"Error loading images: {image_loading_result[1]}")
    else:
        image_loading_result = None


    
    user_proxy = CapAgent(
        name="Assistant",
        prompt_generator = prompt_generator,
        executor=executor,
        code_execution_config={
            "use_docker": False
        },
        is_termination_msg=checks_terminate_message,
        parser=parser
    )

    
    # The user proxy agent is used for interacting with the assistant agent
    # and executes tool calls.
    
    assistant = ConversableAgent(
        name="planner",
        llm_config={
            "config_list": [
        {
            "model": "deepseek/deepseek-r1-0528:free",
            "api_key": os.environ["OPENROUTER_API_KEY"],
            "base_url": "https://openrouter.ai/api/v1",
            "price": [0, 0]
        },
        {
            "model": "qwen/qwen2.5-7b-instruct:free",
            "api_key": os.environ["OPENROUTER_API_KEY"],
            "base_url": "https://openrouter.ai/api/v1",
            "price": [0, 0]
        },
        {
            "model": "mistralai/mistral-7b-instruct:free",
            "api_key": os.environ["OPENROUTER_API_KEY"],
            "base_url": "https://openrouter.ai/api/v1",
            "price": [0, 0]
        }
    ]

        },
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
        is_termination_msg = lambda x: False,
        system_message=ASSISTANT_SYSTEM_MESSAGE,
    )

    
    chat_result, messages = user_proxy.initiate_chat(
        assistant, 
        message=user_query, 
        n_image=len(image_paths) if image_paths is not None else 0
    )

    return chat_result, messages


if __name__ == "__main__":
    
    user_query = """Create a detailed description of the image, focusing on the central figure seated in an ornate throne, wearing an elaborate crown and regal robes. Highlight the presence of clergy in ceremonial attire standing nearby, emphasizing their roles in the event. Note the richly decorated surroundings, including the vibrant colors and intricate patterns. Mention the distinguished guests in formal attire seated in the background, adding context to the ceremonial setting. The description should be informative and concise, around 100 words, with a formal and respectful tone.

Constraints:
Semantic Constraints: Convey a sense of grandeur and tradition.
Format Constraints: Provide a single, structured paragraph. Aim for around 100 words.
Content Constraints: Focus on the central figure, clergy, and ceremonial elements.
Avoid unrelated or speculative details.
Search Constraints: This seems to be a special moment in history, please search it on web."""
    chat_result, messages = run_agent(user_query=user_query, working_dir=".", image_paths=["assets/figs/charles_on_the_throne.png"])
    #from IPython import embed; embed()
