import os
import copy
import numpy as np

from PIL import Image
from nltk.tokenize import word_tokenize, sent_tokenize
from serpapi import GoogleSearch

from capagent.config import (
    DETECTION_CLIENT_HOST, 
    DEPTH_CLIENT_HOST, 
    IMAGE_SERVER_DOMAIN_NAME
)
from capagent.chat_models.client import llm_client, mllm_client
from capagent.utils import encode_pil_to_base64
from gradio_client import Client, file
from pprint import pprint


try:
    detection_client = Client(DETECTION_CLIENT_HOST)
    print("Detection client is listening on port 8080.")
except Exception as e:
    print("Detection client is not working properly. Tools related to detection will not work.")
    detection_client = None

try:
    depth_client = Client(DEPTH_CLIENT_HOST)
    print("Depth client is listening on port 8081.")
except Exception as e:
    print("Depth client is not working properly. Tools related to depth will not work.")
    depth_client = None


class ImageData:
    """
    A class to store the image and its URL for temporary use.
    """

    def __init__(self, image: Image.Image, image_url: str, local_path: str):
        """
        Args:
            image (PIL.Image.Image): The image to store
            image_url (str): The image URL to store
        """
        self.image: Image.Image = image
        self.image_url: str = image_url
        self.local_path: str = local_path


def visual_question_answering_image(query: str, image: Image.Image, show_result: bool = True) -> str:
    """
    Answer a question about a PIL.Image object directly (without ImageData wrapper).
    """
    messages = [
        {
            "role": "user", 
            "content": [
                {
                    'type': 'image_url', 
                    'image_url': {
                        'url': f"data:image/jpeg;base64,{encode_pil_to_base64(image)}"
                    }
                },
                {'type': 'text', 'text': query}
            ]
        }
    ]

    result = mllm_client.chat_completion(messages)
    if show_result:
        print(f"Answer to the question: {result}")

    return result


'''def crop_object_region_image(image: Image.Image, object_name: str, save_path="./.tmp/crop_image.png") -> Image.Image:
    """
    Crop a detected object region from a PIL.Image object.
    Returns the cropped PIL.Image.
    """
    # Run object detection on the raw image
    _, result_json = detection_client.predict(file(save_path), object_name, 0.3, 0.3)
    bbox = result_json['bboxes'][0]  # cxcywh, relative
    width, height = image.size

    # Convert to absolute coordinates
    abs_bbox = [
        int((bbox[0] - bbox[2] / 2) * width), 
        int((bbox[1] - bbox[3] / 2) * height), 
        int((bbox[0] + bbox[2] / 2) * width), 
        int((bbox[1] + bbox[3] / 2) * height)
    ]

    crop_image = image.crop(abs_bbox)
    crop_image.save(save_path)
    return crop_image'''


'''def counting_object_image(image: Image.Image, object_name: str = None, show_result: bool = True) -> int:
    """
    Count the number of occurrences of an object in the PIL.Image.
    """
    _, result_json = detection_client.predict(file("./.tmp/temp_image.png"), object_name, 0.3, 0.3)
    count = len(result_json["phrases"])
    if show_result:
        print(f"There are {count} {object_name}(s) in the image.")
    return count
'''

'''def spatial_relation_of_objects_image(image: Image.Image, objects: list[str], show_result: bool = True) -> str:
    """
    Describe the spatial relation of objects in a PIL.Image directly.
    """
    position_list = []

    _, grayscale_depth_map, _ = depth_client.predict(file("./.tmp/temp_image.png"), api_name="/on_submit")
    depth_map = Image.open(grayscale_depth_map).convert("L")
    depth_map = np.array(depth_map) / 255.0  # normalize

    for object_name in objects:
        _, result_json = detection_client.predict(file("./.tmp/temp_image.png"), object_name, 0.3, 0.3)
        for bbox, phrase in zip(result_json['bboxes'], result_json['phrases']):
            rel_bbox = [(bbox[0] - bbox[2] / 2), (bbox[1] - bbox[3] / 2), (bbox[0] + bbox[2] / 2), (bbox[1] + bbox[3] / 2)]
            abs_bbox = [int(rel_bbox[0]*image.width), int(rel_bbox[1]*image.height), int(rel_bbox[2]*image.width), int(rel_bbox[3]*image.height)]
            depth_value = depth_map[abs_bbox[1]:abs_bbox[3], abs_bbox[0]:abs_bbox[2]].mean()
            position_list.append({"object": object_name, "relative_bbox": rel_bbox, "phrase": phrase, "relative_depth_value": depth_value})

    pose_info_str = "\n".join([f"{p['object']}, bbox: {p['relative_bbox']}, depth: {p['relative_depth_value']}" for p in position_list])

    llm_messages = [
        {"role": "system", "content": "You are a helpful assistant describing spatial relationships of objects in an image."},
        {"role": "user", "content": f"{pose_info_str}\nDescribe the spatial relation of the objects."}
    ]

    result = llm_client.chat_completion(llm_messages)
    if show_result:
        print(f"Spatial relation of objects:\n{result}")

    return result'''



'''def visual_question_answering(query: str, image_data: ImageData, show_result: bool = True) -> str:
    """
    Answer a question about the image.
    
    Args:
        query (str): The question to answer
        image_data (ImageData): The image data to answer the question about
        show_result (bool): Whether to print the result
    Returns:
        str: The answer to the question
    """
    messages = [
        {
            "role": "user", 
            "content": [
                {
                    'type': 'image_url', 
                    'image_url': {
                        'url': f"data:image/jpeg;base64,{encode_pil_to_base64(image_data.image)}"
                    }
                },
                {'type': 'text', 'text': query}
            ]
        }
    ]

    result = mllm_client.chat_completion(messages)
    if show_result:
        print(f"Answer to the question: {result}")

    return result
  '''  

def count_words(caption: str, show_result: bool = True) -> int:
    """
    Count the number of words in the input string.
    
    Args:
        caption (str): The input string to count the words
        show_result (bool): Whether to print the result
        
    Returns:
        int: The number of words in the input string
    """
    if show_result:
        print(f"Now the number of words in the caption is: {len(word_tokenize(caption))}.")

    return len(word_tokenize(caption))


def count_sentences(caption: str, show_result: bool = True) -> int:
    """
    Count the number of sentences in the input string.

    Args:
        caption (str): The input string to count the sentences
        show_result (bool): Whether to print the result
        
    Returns:
        int: The number of sentences in the input string
    """
    sentences = sent_tokenize(caption)
    if show_result:
        print(f"The number of sentences in the caption is: {len(sentences)}.")

    return len(sentences)


def shorten_caption(caption: str, max_words: int = None, max_sentences: int = None, show_result: bool = True) -> str:
    """
    Shorten the caption within the max length while maintaining key information.
    Before calling this function, you should call the count_words or count_sentences function to check if the caption is already short enough.
    Args:
        caption (str): The original caption text to be shortened
        max_words (int): Maximum number of words allowed in the shortened caption
        max_sentences (int): Maximum number of sentences allowed in the shortened caption
        show_result (bool): Whether to print the result
    
    Returns:
        str: A shortened version of the input caption that respects the word limit
    """

    system_prompt = """You are helpful assistant. You are good at shortening the image caption. Each time the user provides a caption and the max length, you can help to shorten the caption to the max length.

    Note:
    - You can change the length of the caption by first delete unnecessary words.
    - You should keep the original sentiment and descriptive perspective of the caption.
    - You should keep the original meaning of the caption.
    """
    assert max_words is not None or max_sentences is not None, "Either max_words or max_sentences should be provided."
    
    length_constrain = f"Max length: {max_words} words." if max_words is not None else f"Max length: {max_sentences} sentences."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Caption: {caption}. {length_constrain}. Directly output the shortened caption without any other words."} 
    ]
    result = llm_client.chat_completion(messages)

    if max_words is not None:
        w_count = count_words(result, show_result=False)
        while w_count > max_words:
            messages += [{"role": "assistant", "content": f"Caption: {result}"}]
            messages += [{"role": "user", "content": f"The length of the caption ({w_count} words) is still longer than the max length ({max_words} words). Please shorten the caption to the max length."}]
            result = llm_client.chat_completion(messages)
            w_count = count_words(result, show_result=False)
    
    elif max_sentences is not None:
        s_count = count_sentences(result, show_result=False)
        while s_count > max_sentences:
            messages += [{"role": "assistant", "content": f"Caption: {result}"}]
            messages += [{"role": "user", "content": f"The number of sentences in the caption ({s_count} sentences) is still longer than the max length ({max_sentences} sentences). Please shorten the caption to the max length."}]
            result = llm_client.chat_completion(messages)
            s_count = count_sentences(result, show_result=False)
    
    if show_result:
        print(f"Shortened caption: {result}")

    return result

def change_caption_sentiment(caption: str, sentiment: str, show_result: bool = True) -> str:
    """
    Transfer the caption to the specified sentiment.
    
    Args:
        caption (str): The original caption text to be transferred
        sentiment (str): The desired sentiment for the caption
        show_result (bool): Whether to print the result
    
    Returns:
        str: The caption with the transferred sentiment

    This function will automatically print the result by setting show_result to True, with the transferred caption and the number of words in the caption.
    """

    user_input = {"role": "user", "content": f"Caption: {caption}. Please change the sentiment of the caption to {sentiment}. Directly output the transferred caption without any other words."}
    messages = [user_input]
    result = llm_client.chat_completion(messages)
    if show_result:
        print(f"Transferred caption: {result}")

    return result


def extend_caption(image: Image.Image, caption: str, iteration: int = 1, show_result: bool = True, local_path: str = None) -> str:
    """
    Extend a caption to include more details using multiple iterations of question-answering about the image.

    Args:
        image (PIL.Image.Image): The image to analyze.
        caption (str): The base caption to extend.
        iteration (int): Number of iterations to ask and answer questions.
        show_result (bool): Whether to print debug info and final caption.
        local_path (str, optional): Local path to the image (for logging or reference).

    Returns:
        str: The extended caption.
    """
    system_prompt = (
        "You are a helpful assistant that can help users extend a caption to include more details. "
        "You can ask questions about the image and use the answers to enhance the caption."
    )

    llm_messages = [{"role": "system", "content": system_prompt}]
    llm_messages.append({"role": "user", "content": f"Caption: {caption}. Please generate a question to extend the caption."})

    for _ in range(iteration):
        # Step 1: Ask the LLM to generate a question
        question = llm_client.chat_completion(llm_messages)

        # Step 2: Use multi-modal client to answer the question using the image
        mllm_message = [{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encode_pil_to_base64(image)}"
                    }
                },
                {"type": "text", "text": question}
            ]
        }]

        answer = mllm_client.chat_completion(mllm_message)

        # Step 3: Add question-answer to LLM context for next iteration
        llm_messages += [
            {"role": "assistant", "content": f"Question: {question}"},
            {"role": "user", "content": f"Answer: {answer}. Please generate a new question."}
        ]

    # Step 4: Finally, generate the extended caption
    llm_messages.append({
        "role": "user",
        "content": f"Please extend the caption according to the questions and answers. Caption: {caption}. Directly output the extended caption."
    })

    result = llm_client.chat_completion(llm_messages)

    if show_result:
        print(f"Extended caption: {result}")
        count_words(result, show_result=True)
        count_sentences(result, show_result=True)

    return result



'''def extend_caption(image_data: ImageData, caption: str, iteration: int, show_result: bool = True) -> str:
    """
    Call this function when you need to extend the caption to include more details. 
    Before calling this function, you should call the count_words or count_sentences function to check if the caption is already detailed enough.
    Args:
        image_data (ImageData): The image data to extend the caption
        caption (str): The caption to extend
        iteration (int): The number of iterations to ask and answer questions
        show_result (bool): Whether to print the result
    
    Returns:
        str: The extended caption
        
    This function will automatically print the result by setting show_result to True, with the extended caption and the number of words in the caption.
    """
    system_prompt = "You are a helpful assistant that can help users to extend the caption to include more details. You can ask one or more questions about the image to the user. The user will answer each question. After you get all the answers, you can add more information to the caption according to the answers."
    
    user_input = {"role": "user", "content": f"Caption: {caption}. Please generate a question to extend the caption."}
    llm_messages = [{"role": "system", "content": system_prompt}, user_input]
    
    for _ in range(iteration):
        question = llm_client.chat_completion(llm_messages)
        mllm_message = [{
            "role": "user", 
            "content": [
                {
                    'type': 'image_url', 
                    'image_url': {
                        'url': f"data:image/jpeg;base64,{encode_pil_to_base64(image_data.image)}"
                    }
                },
                {'type': 'text', 'text': question}
            ]
        }]
        answer = mllm_client.chat_completion(mllm_message)
        llm_messages += [
            {"role": "assistant", "content": f"Question: {question}"}, 
            {"role": "user", "content": f"Answer: {answer}. Please generate a new question."}
        ]
    
    llm_messages += [{"role": "user", "content": f"Please extend the caption according to the questions and answers. Caption: {caption}. Directly output the extended caption without any other words."}]
    result = llm_client.chat_completion(llm_messages)
    
    if show_result:
        print(f"Extended caption: {result}.")
        count_words(result, show_result=True)
        count_sentences(result, show_result=True)

    return result
'''

def add_keywords_to_caption(caption: str, keywords: list[str], show_result: bool = True) -> str:
    """
    Call this function when you need to add keywords to the caption.

    Args:
        caption (str): The original caption of the image
        keywords (list[str]): The keywords to add to the caption
        show_result (bool): Whether to print the result

    Returns:
        str: The caption with the added keywords
            
    This function will automatically print the result by setting show_result to True, with the added keywords and the number of words in the caption.
    """
    system_prompt = "You are a helpful assistant that can help users to add keywords to the caption. Please ensure the readability of the output caption."
    user_input = {"role": "user", "content": f"Please add these keywords: {keywords} to the caption:\n{caption}. Directly output your answer without any other words."}
    messages = [{"role": "system", "content": system_prompt}, user_input]
    result = llm_client.chat_completion(messages)

    if show_result:
        print(f"Caption with keywords: {result}")

    return result

def google_search(query: str, show_result: bool = True, top_k: int = 5) -> str:
    """
    Call this function when you need to search the query on Google.
    
    Args:
        query (str): The query to search
        show_result (bool): Whether to print the result
        top_k (int): The number of results to show

    Returns:
        str: The search result

    This function will automatically print the search result by setting show_result to True, with the title, snippet, snippet highlighted words, source, and the link of the result.
    """

    params = {
        "q": query,
        "location": "Austin, Texas, United States",
        "hl": "en",
        "gl": "us",
        "google_domain": "google.com",
        "api_key": os.getenv("SERP_API_KEY")
    }

    search = GoogleSearch(params)
    results = search.get_dict()

    organic_results = results.get("organic_results", None)
    
    if results is None and show_result:
        search_result = "No results found"
    else:
        search_result = f"Google Search Result of {query}:"
        for i, result in enumerate(organic_results[:top_k]):
            search_result += "\n" + "-" * 10 + f"Result {i + 1}" + "-" * 10
            search_result += f"\nTitle: {result.get('title', 'N/A')}"
            search_result += f"\nSnippet: {result.get('snippet', 'N/A')}"
            search_result += f"\nSnippet highlighted words: {result.get('snippet_highlighted_words', 'N/A')}"
            search_result += f"\nSource: {result.get('source', 'N/A')}\n"

    if show_result:
        print(search_result)

    return search_result



def google_lens_search(image_data: ImageData, show_result: bool = True, top_k: int = 10) -> str:
    """
    Call this function when you need to search the similar images information on Google Lens. 

    Args:
        image_data (ImageData): The image data to search the similar images on Google Lens
        show_result (bool): Whether to print the result
        top_k (int): The number of results to show

    Returns:
        str: The search result

    This function will automatically print the search result by setting show_result to True, with the title of each similar image.
    """

    params = {
        "engine": "google_lens",
        "url": image_data.image_url,
        "api_key": os.getenv("SERP_API_KEY"),
        "hl": "en",
        "country": "US",
        "no_cache": True
    }

    try: 
        search = GoogleSearch(params)
        results = search.get_dict()
        #print("Google Lens API raw results:", results)  # <--- Add this line
        visual_matches = results.get("visual_matches", [])

        titles = [v_match["title"] for v_match in visual_matches[:top_k]]
        search_result = "Google Lens Image Search Result:"

        for i, title in enumerate(titles):
            search_result += "\n" + "-" * 10 + f"Result {i + 1}" + "-" * 10
            search_result += f"\nTitle: {title}"

    except Exception as e:
        print("Google Lens Search Exception:", e)  # <--- Add this line
        search_result = "This tool is experiencing problems and is not working properly"

    if show_result:
        print(search_result)

    return search_result


def crop_object_region(image_data: ImageData, object: str) -> str:
    """
    Call this function when you need to crop the object region in the image.

    Args:
        image_data (ImageData): The image data to crop the object region
        object (str): The object to crop
    
    Returns:
        ImageData: The image data contains the cropped object region image
    """ 

    _, result_json = detection_client.predict(file(image_data.local_path), object, 0.3, 0.3)  
    bbox = result_json['bboxes'][0]  # cxcywh, relative position
    width, height = image_data.image.size
    
    # turn to absolute position
    bbox = [
        int((bbox[0] - bbox[2] / 2) * width), 
        int((bbox[1] - bbox[3] / 2) * height), 
        int((bbox[0] + bbox[2] / 2) * width), 
        int((bbox[1] + bbox[3] / 2) * height)
    ]

    # crop the image
    crop_image = copy.deepcopy(image_data.image).crop((bbox[0], bbox[1], bbox[2], bbox[3]))
    crop_image.save("./.tmp/crop_image.png")
    crop_image_data = ImageData(image=crop_image, image_url=f"{IMAGE_SERVER_DOMAIN_NAME}/.tmp/crop_image.png", local_path="./.tmp/crop_image.png")
    
    return crop_image_data


def counting_object(image_data: ImageData, object: str = None, show_result: bool = True):
    """
    Call this function when you need to count the number of the object in the image.

    Args:
        image_data (ImageData): The image data to count the object
        object (str): The object to count
        show_result (bool): Whether to print the result
    
    This function will automatically print the the number of the object by setting show_result to True.
    """

    _, result_json = detection_client.predict(file(image_data.local_path), object, 0.3, 0.3)
    if show_result:
        print(f'There are {len(result_json["phrases"])} {result_json["phrases"][0]} in the image.')

def spatial_relation_of_objects(image_data: ImageData, objects: list[str], show_result: bool = True) -> str:
    """
    Call this function when you need to know the depth value and spatial relation of the objects in the image.

    Args:
        image_data (ImageData): The image data to get the depth value and spatial relation of the objects
        show_result (bool): Whether to print the result
        objects (list[str]): The objects to get the depth value and spatial relation

    Returns:
        str: The spatial relation of the objects
    """

    position_list = []

    _, grayscale_depth_map, _ = depth_client.predict(file(image_data.local_path), api_name="/on_submit")

    depth_map = Image.open(grayscale_depth_map).convert("L")
    depth_map.save("./.tmp/depth_map.png")
    depth_map = np.array(depth_map)
    # normalize the depth map to 0-1
    depth_map = depth_map / 255.0

    assert objects is not None, "Objects are not specified."
    for object in objects:
        _, result_json = detection_client.predict(file(image_data.local_path), object, 0.3, 0.3)

        for bbox, phrase in zip(result_json['bboxes'], result_json['phrases']):

            relative_bbox = [
                (bbox[0] - bbox[2] / 2), 
                (bbox[1] - bbox[3] / 2), 
                (bbox[0] + bbox[2] / 2), 
                (bbox[1] + bbox[3] / 2)
            ]

            absolute_bbox = [
                int(relative_bbox[0] * image_data.image.width), 
                int(relative_bbox[1] * image_data.image.height), 
                int(relative_bbox[2] * image_data.image.width), 
                int(relative_bbox[3] * image_data.image.height)
            ]

            # from IPython import embed; embed()

            # calulate the average depth value of the object
            object_depth_value = depth_map[absolute_bbox[1]:absolute_bbox[3], absolute_bbox[0]:absolute_bbox[2]].mean()
            position_list.append({"object": object, "relative_bbox": relative_bbox, "phrase": phrase, "relative_depth_value": object_depth_value})

    # gain depth map of the image
    pose_info_str = ""
    for object_pose_info in position_list:
        pose_info_str += f"{object_pose_info['object']}, bounding box: {object_pose_info['relative_bbox']}, depth value: {object_pose_info['relative_depth_value']}\n"

    print(pose_info_str)

    llm_messages = [
        {"role": "system", "content": "You are a helpful assistant that aids users in understanding the spatial relationships of objects in an image. You can access the average depth value (ranging from 0 to 1, where 0 is shallow and 1 is deep) for each object region and their bounding box coordinates (x1, y1, x2, y2), also ranging from 0 to 1, with (0, 0) as the top-left corner. Use descriptive language to explain the positional relationships without including precise values."},
        
        {"role": "user", "content": f"{pose_info_str}\nPlease describe the spatial relation of the objects in the image."}
    ]
    result = llm_client.chat_completion(llm_messages)

    # from IPython import embed; embed()

    if show_result:
        print(f"Spatial relation of the objects:\n{result}")