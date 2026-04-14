from capagent.chat_models.client import mllm_client
from capagent.utils import encode_pil_to_base64
from capagent.tools import google_search, google_lens_search, ImageData
from PIL import Image


INSTRUCTION_AUGMENTATION_SYSTEM_MESSAGE = """You are an intelligent assistant that generates professional caption instructions based on a given image's visual content and a simple user input. Your task is to analyze the visual content of the image and transform the user's simple instruction into a detailed, professional caption instruction.

You can add one of the following constrains to the instruction if you want. Here is some constraint dimension you can refer to:
1. Keywords or phrases: You can add suitable keywords or phrases to the instruction according to the image and original description. E.g., please include the words: "Boeing 737", "a long wings" in the description.
2. Sentiment: You can add suitable sentiment constraints to the instruction according to the image and original description. E.g., describe the image with a happy sentiment.
3. Length: You can add length constraints to the instruction and generate the corresponding description. E.g., using 10 words to describe the image.
4. Focus content: You can add focus content constraints to the instruction according to the image and original description. E.g., focus on the material of the vase.
5. Format: You can add format constraints (single paragraph / markdown / html) to the instruction according to the image and original description. E.g., First, summary the image in a single paragraph, then use bullet points to describe the color and material of the car in the image.
6. Viewpoint: You can add viewpoint constraints to the instruction according to the image and original description. E.g., describe the image from the middle person's perspective.
7. Genre: You can add genre constraints to the instruction according to the image and original description. E.g., describe the image in the style of a children's book; Describe the image in the style of a poem; Describe the image in the style of a news report; Describe the image in the style of a travel blog post; 
...

NOTE: 
- Ensure you incorporate essential constraints from the original user instruction. 
- Adapt the instruction to the given visual content, user intent, and image characteristics.
- You should design a suitable format for the caption, according to other constraints and visual content to improve the readability of the caption.
- The professional instruction should be start with "Please describe the image according to the following instructions:", then format each constraint in a new line.
- Directly output the instruction without any other words.
- The format of the instruction should be suitable for the user to read and understand. For example, when there are multiple details of same object, you can ask the captioner to write each aspect of the object using bullet points.
"""


SEARCH_ASSISTANT_SYSTEM_MESSAGE = """You are an intelligent assistant that can search on the web. 
The user will provide you an image and image search result on the web.
You need to generate a keywords list for further search on the web. Such information will be used to generate a more accurate instruction to guide the image captioning.
"""


class InstructionAugmenter:

    EXAMPLES = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encode_pil_to_base64(Image.open('data/cia_examples/0.png').convert('RGB'))}"
                    }
                },
                {
                    "type": "text",
                    "text": "User instruction: Please describe the image within 100 words. \nPlease generate a professional instruction based on user instruction. Directly output the instruction without any other words."
                }
            ]
        },
        {"role": "assistant", "content": open("data/cia_examples/0.txt", "r").read()},
    ]


    def generate_complex_instruction(self, image, image_url: str, query: str, is_search: bool, timeout=20):
        """
        Generates a complex instruction from an image and/or its URL, plus a user query.
        - image: PIL.Image.Image object (can be None if only URL is available)
        - image_url: Direct image URL (can be None if only image is available)
        - query: user query text
        - is_search: whether to run Google Lens + web search pipeline
        """

        messages = [
            {"role": "system", "content": INSTRUCTION_AUGMENTATION_SYSTEM_MESSAGE}
        ]

        if is_search:
            # ✅ Always send to Google Lens
            uploaded_url = image_url
            local_path = None

            if not uploaded_url:
                if image:
                    local_path = ".tmp/search_image.png"
                    image.save(local_path)
                    uploaded_url = upload_to_imgbb(local_path)
                else:
                    raise ValueError("Either image or image_url must be provided for search mode.")

            image_data = ImageData(
                image=image,
                image_url=uploaded_url,
                local_path=local_path
            )
            print(image_data.image_url)
            image_search_result = google_lens_search(image_data)  # Always called
            print("##############################################")
            print(image_search_result)
            # Step 2: Generate keywords (NO IMAGE sent to mllm_client)
            search_prompt = (
                f"Here is the image search result:\n{image_search_result}\n"
                "Based on this, please output no more than 5 most informative keywords or phrases."
            )
            search_messages = [
                {"role": "system", "content": SEARCH_ASSISTANT_SYSTEM_MESSAGE},
                {"role": "user", "content": search_prompt}
            ]
            print("DEBUG search prompt:",search_prompt)
            print("DEBUG search messages:",search_messages)
            search_keywords = mllm_client.chat_completion(search_messages, timeout=timeout)
            print(f"search_keywords: {search_keywords}")
            if search_keywords is None:
                print("WARNING: search_keywords is None!")
                search_keywords = ""  # fallback
            # Step 3: Google Search using keywords
            keywords_search_result = google_search(search_keywords)
            print("DEBUG: keywords_search_result =", keywords_search_result)

            # Step 4: Summarize Google Search results
            summary_prompt = (
                f"Here is the search result for keywords:\n{keywords_search_result}.\n\n"
                "Now please summarize it."
            )
            summary_messages = [
                {"role": "system", "content": SEARCH_ASSISTANT_SYSTEM_MESSAGE},
                {"role": "user", "content": summary_prompt}
            ]
            search_information_summary = mllm_client.chat_completion(summary_messages, timeout=timeout)
            print(f"search_information_summary: {search_information_summary}")
            if search_information_summary is None:
                print("WARNING: search_information_summary is None!")
                search_information_summary = "No summary generated"
            # Step 5: Generate final instruction
            instruction_prompt = (
                f"User instruction: {query}. \n"
                f"Based on the above search information: {search_information_summary}, "
                "please generate a professional instruction. Directly output the instruction without any other words."
            )
            final_messages = [
                {"role": "system", "content": INSTRUCTION_AUGMENTATION_SYSTEM_MESSAGE},
                {"role": "user", "content": instruction_prompt}
            ]
            final_instruction = mllm_client.chat_completion(final_messages, timeout=timeout)
            print("DEBUG: final_instruction =", final_instruction)
            if final_instruction is None:
                print("WARNING: final_instruction is None!")
                final_instruction = "Error: No instruction generated"
            return mllm_client.chat_completion(final_messages, timeout=timeout)

        else:
            # No search path
            user_content = (
                f"User instruction: {query}. Please generate a professional instruction based on user instruction. "
                "Directly output the instruction without any other words."
            )
            messages.append({"role": "user", "content": user_content})

            return mllm_client.chat_completion(messages, timeout=timeout)

    

                    


if __name__ == "_main_":
    ia = InstructionAugmenter()
    print(ia.generate_complex_instruction(
        Image.open("assets/figs/trump_assassination.png").convert("RGB"),
        "Please describe the image.",
        is_search=True
    ))


    '''def generate_complex_instruction(self, image, image_url: str, query: str, is_search: bool, timeout=20):
        """
        Generates a complex instruction from an image and/or its URL, plus a user query.
        - image: PIL.Image.Image object (can be None if only URL is available)
        - image_url: Direct image URL (can be None if only image is available)
        - query: user query text
        - is_search: whether to run Google Lens + web search pipeline
        """

        messages = [
            {"role": "system", "content": INSTRUCTION_AUGMENTATION_SYSTEM_MESSAGE}
        ]

        if is_search:
            # Step 1: Google Lens image search
            if getattr(mllm_client, "supports_vision", False):
                uploaded_url = image_url
                local_path = None

                if not uploaded_url:
                    # No URL provided, must upload the image
                    if image:
                        local_path = ".tmp/search_image.png"
                        image.save(local_path)
                        uploaded_url = upload_to_imgbb(local_path)
                    else:
                        raise ValueError("Either image or image_url must be provided for search mode.")

                image_data = ImageData(
                image=image,
                image_url=uploaded_url,
                local_path=local_path
                )
                image_search_result = google_lens_search(image_data)
            else:
                image_search_result = "(Image search skipped: model is text-only)"

            # Step 2: Generate keywords from image search result
            search_prompt = (
                f"Here is the image search result:\n{image_search_result}\n"
                "Based on this, please output no more than 5 most informative keywords or phrases."
            )
            search_messages = [
                {"role": "system", "content": SEARCH_ASSISTANT_SYSTEM_MESSAGE},
                {"role": "user", "content": search_prompt}
            ]
            search_keywords = mllm_client.chat_completion(search_messages, timeout=timeout)
            print(f"search_keywords: {search_keywords}")

            # Step 3: Google Search using keywords
            keywords_search_result = google_search(search_keywords)

            # Step 4: Summarize Google Search results
            summary_prompt = (
            f"Here is the search result for keywords:\n{keywords_search_result}.\n\n"
            "Now please summarize it."
            )
            summary_messages = [
            {"role": "system", "content": SEARCH_ASSISTANT_SYSTEM_MESSAGE},
            {"role": "user", "content": summary_prompt}
            ]
            search_information_summary = mllm_client.chat_completion(summary_messages, timeout=timeout)
            print(f"search_information_summary: {search_information_summary}")

            # Step 5: Generate final instruction
            instruction_prompt = (
            f"User instruction: {query}. \n"
            f"Based on the above search information: {search_information_summary}, "
            "please generate a professional instruction. Directly output the instruction without any other words."
            )
            final_messages = [
            {"role": "system", "content": INSTRUCTION_AUGMENTATION_SYSTEM_MESSAGE},
            {"role": "user", "content": instruction_prompt}
            ]

            return mllm_client.chat_completion(final_messages, timeout=timeout)

        else:
            # No search path
            user_content = (
            f"User instruction: {query}. Please generate a professional instruction based on user instruction. "
            "Directly output the instruction without any other words."
            )
            messages.append({"role": "user", "content": user_content})

            return mllm_client.chat_completion(messages, timeout=timeout)
'''

    '''def generate_complex_instruction(self, image, query: str, is_search: bool, timeout=20):
        #from gradio_demo import upload_to_imgbb

        messages = [
            {"role": "system", "content": INSTRUCTION_AUGMENTATION_SYSTEM_MESSAGE}
        ]

        if is_search:
            # Step 1: Google Lens image search (only if vision supported)
            if getattr(mllm_client, "supports_vision", False):
                image.save(".tmp/search_image.png")
                uploaded_url = upload_to_imgbb(".tmp/search_image.png")
                image_data = ImageData(
                    image,
                    image_url=uploaded_url,  # this might be unused, consider removing or fix URL

                    local_path=".tmp/search_image.png"
                )
                image_search_result = google_lens_search(image_data)
            else:
                image_search_result = "(Image search skipped: model is text-only)"

            # Step 2: Generate keywords from image search result using LLM
            search_prompt = (
                f"Here is the image search result:\n{image_search_result}\n"
                "Based on this, please output no more than 5 most informative keywords or phrases."
            )
            search_messages = [
                {"role": "system", "content": SEARCH_ASSISTANT_SYSTEM_MESSAGE},
                {"role": "user", "content": search_prompt}
            ]
            search_keywords = mllm_client.chat_completion(search_messages, timeout=timeout)
            print(f"search_keywords: {search_keywords}")

            # Step 3: Use keywords to perform a Google Search (text only)
            keywords_search_result = google_search(search_keywords)

            # Step 4: Summarize Google Search results with LLM
            summary_prompt = (
                f"Here is the search result for keywords:\n{keywords_search_result}.\n\n"
                "Now please summarize it."
            )
            summary_messages = [
                {"role": "system", "content": SEARCH_ASSISTANT_SYSTEM_MESSAGE},
                {"role": "user", "content": summary_prompt}
            ]
            search_information_summary = mllm_client.chat_completion(summary_messages, timeout=timeout)
            print(f"search_information_summary: {search_information_summary}")

            # Step 5: Generate final professional instruction based on user query and search summary
            instruction_prompt = (
                f"User instruction: {query}. \n"
                f"Based on the above search information: {search_information_summary}, "
                "please generate a professional instruction. Directly output the instruction without any other words."
            )
            final_messages = [
                {"role": "system", "content": INSTRUCTION_AUGMENTATION_SYSTEM_MESSAGE},
                {"role": "user", "content": instruction_prompt}
            ]

            return mllm_client.chat_completion(final_messages, timeout=timeout)

        else:
            # If not a search, generate instruction from user query only
            user_content = (
                f"User instruction: {query}. Please generate a professional instruction based on user instruction. "
                "Directly output the instruction without any other words."
            )
            messages.append({"role": "user", "content": user_content})

            return mllm_client.chat_completion(messages, timeout=timeout)
'''


'''from capagent.chat_models.client import mllm_client
from capagent.utils import encode_pil_to_base64
from capagent.tools import google_search, google_lens_search, ImageData
from PIL import Image



INSTRUCTION_AUGMENTATION_SYSTEM_MESSAGE = f"""You are an intelligent assistant that generates professional caption instructions based on a given image's visual content and a simple user input. Your task is to analyze the visual content of the image and transform the user's simple instruction into a detailed, professional caption instruction.

You can add one of the following constrains to the instruction if you want. Here is some constraint dimension you can refer to:
1. Keywords or phrases: You can add suitable keywords or phrases to the instruction according to the image and original description. E.g., please include the words: "Boeing 737", "a long wings" in the description.
2. Sentiment: You can add suitable sentiment constraints to the instruction according to the image and original description. E.g., describe the image with a happy sentiment.
3. Length: You can add length constraints to the instruction and generate the corresponding description. E.g., using 10 words to describe the image.
4. Focus content: You can add focus content constraints to the instruction according to the image and original description. E.g., focus on the material of the vase.
5. Format: You can add format constraints (single paragraph / markdown / html) to the instruction according to the image and original description. E.g., First, summary the image in a single paragraph, then use bullet points to describe the color and material of the car in the image.
6. Viewpoint: You can add viewpoint constraints to the instruction according to the image and original description. E.g., describe the image from the middle person's perspective.
7. Genre: You can add genre constraints to the instruction according to the image and original description. E.g., describe the image in the style of a children's book; Describe the image in the style of a poem; Describe the image in the style of a news report; Describe the image in the style of a travel blog post; 
...


NOTE: 
- Ensure you incorporate essential constraints from the original user instruction. 
- Adapt the instruction to the given visual content, user intent, and image characteristics.
- You should design a suitable format for the caption, according to other constraints and visual content to improve the readability of the caption.
- The professional instruction should be start with "Please describe the image according to the following instructions:", then format each constraint in a new line.
- Directly output the instruction without any other words.
- The format of the instruction should be suitable for the user to read and understand. For example, when there are multiple details of same object, you can ask the captioner to write each aspect of the object using bullet points.
"""


SEARCH_ASSISTANT_SYSTEM_MESSAGE = f"""You are an intelligent assistant that can search on the web. 
The user will provide you an image and image search result on the web.
You need to generate a keywords list for further search on the web. Such information will be used to generate a more accurate instruction to guide the image captioning.
"""


class InstructionAugmenter:

    EXAMPLES = [
        {
            "role": "user", "content": 
            [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encode_pil_to_base64(Image.open('data/cia_examples/0.png').convert('RGB'))}"
                    }
                },
                {
                    "type": "text",
                    "text": "User instruction: Please describe the image within 100 words. \nPlease generate a professional instruction based on user instruction. Directly output the instruction without any other words."
                }
            ]
        },
        {"role": "assistant", "content": open("data/cia_examples/0.txt", "r").read()},
    ]

    def generate_complex_instruction(self, image, query: str, is_search: bool, timeout=20):

    messages = [
        {"role": "system", "content": INSTRUCTION_AUGMENTATION_SYSTEM_MESSAGE}
    ]

    if is_search:
        # Step 1: Do image search via Google Lens ONLY
        if getattr(mllm_client, "supports_vision", False):
            # Save image locally for search
            image.save(".tmp/search_image.png")
            image_data = ImageData(
                image,
                image_url=f"https://i.imgur.com/abcd123.jpg/.tmp/search_image.png",  # this might be unused, consider removing or fix URL
                local_path=".tmp/search_image.png"
            )
            # Run Google Lens image search via SerpAPI
            image_search_result = google_lens_search(image_data)
        else:
            image_search_result = "(Image search skipped: model is text-only)"

        # Step 2: Prepare text prompt for keyword generation based on image search result and user query
        search_prompt = f"Here is the image search result:\n{image_search_result}\n" \
                        f"Based on this, please output no more than 5 most informative keywords or phrases."

        # Step 3: Run LLM to generate keywords from image search result (text only)
        search_messages = [
            {"role": "system", "content": SEARCH_ASSISTANT_SYSTEM_MESSAGE},
            {"role": "user", "content": search_prompt}
        ]
        search_keywords = mllm_client.chat_completion(search_messages, timeout=timeout)
        print(f"search_keywords: {search_keywords}")

        # Step 4: Run normal Google Search (text-only) with generated keywords
        keywords_search_result = google_search(search_keywords)

        # Step 5: Summarize Google Search results with LLM
        summary_prompt = f"Here is the search result for keywords:\n{keywords_search_result}.\n\nNow please summarize it."

        summary_messages = [
            {"role": "system", "content": SEARCH_ASSISTANT_SYSTEM_MESSAGE},
            {"role": "user", "content": summary_prompt}
        ]
        search_information_summary = mllm_client.chat_completion(summary_messages, timeout=timeout)
        print(f"search_information_summary: {search_information_summary}")

        # Step 6: Generate final professional instruction based on user query + search summary
        instruction_prompt = f"User instruction: {query}. \n" \
                             f"Based on the above search information: {search_information_summary}, " \
                             f"please generate a professional instruction. Directly output the instruction without any other words."

        final_messages = [
            {"role": "system", "content": INSTRUCTION_AUGMENTATION_SYSTEM_MESSAGE},
            {"role": "user", "content": instruction_prompt}
        ]

        # Return the final instruction from LLM (text only)
        return mllm_client.chat_completion(final_messages, timeout=timeout)

    else:
        # If not a search, just generate instruction based on user query (and optionally image if supported)
        user_content = f"User instruction: {query}. Please generate a professional instruction based on user instruction. Directly output the instruction without any other words."

        messages.append({"role": "user", "content": user_content})

        return mllm_client.chat_completion(messages, timeout=timeout)

'''
    
'''def generate_complex_instruction(self, image, query: str, is_search: bool, timeout=20):

        messages = [
            {"role": "system", "content": INSTRUCTION_AUGMENTATION_SYSTEM_MESSAGE}
        ]

        if is_search:
            search_messages = [
                {"role": "system", "content": SEARCH_ASSISTANT_SYSTEM_MESSAGE}
            ]

            if getattr(mllm_client, "supports_vision", False):
                # Only do Google Lens search if vision is supported
                image.save(".tmp/search_image.png")
                image_data = ImageData(
                    image,
                    image_url=f"https://i.imgur.com/abcd123.jpg/.tmp/search_image.png",
                    local_path=f".tmp/search_image.png"
                )
                image_search_result = google_lens_search(image_data)

                search_user_content = [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encode_pil_to_base64(image)}"
                        }
                    },
                    {
                        "type": "text",
                        "text": f"Here is the image search result:\n{image_search_result}, now please output the no more than 5 most informative phrases..."
                    }
                ]
            else:
                # Skip Google Lens entirely for text-only models
                image_search_result = "(Image search skipped: model is text-only)"
                search_user_content = [
                    {
                        "type": "text",
                        "text": f"Image search skipped (text-only model). Original query: {query}"
                    }
                ]

            search_messages.append({"role": "user", "content": search_user_content})
            search_keywords = mllm_client.chat_completion(search_messages, timeout=timeout)
            print(f"search_keywords: {search_keywords}")

            keywords_search_result = google_search(search_keywords)

            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Here is the search result for keywords:\n{keywords_search_result}.\n\nNow please summarize it."
                    }
                ]
            })

            search_information_summary = mllm_client.chat_completion(search_messages, timeout=timeout)
            print(f"search_information_summary: {search_information_summary}")

            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": search_information_summary}]
            })
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Now please generate a professional instruction based on user instruction and the above search information. Directly output the instruction without any other words."
                    }
                ]
            })

        else:
            user_content = [
                {
                    "type": "text",
                    "text": f"User instruction: {query}. \nPlease generate a professional instruction based on user instruction. Directly output the instruction without any other words."
                }
            ]

            if getattr(mllm_client, "supports_vision", False):
                user_content.insert(0, {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encode_pil_to_base64(image)}"
                    }
                })

            messages.append({"role": "user", "content": user_content})

        return mllm_client.chat_completion(messages, timeout=timeout)


if _name_ == "_main_":
    ia = InstructionAugmenter()
    print(ia.generate_complex_instruction(
        Image.open("assets/figs/trump_assassination.png").convert("RGB"),
        "Please describe the image.",
        is_search=True
    ))'''


'''def generate_complex_instruction(self, image, query: str, is_search: bool, timeout=20):

        

        messages = [
            {"role": "system", "content": INSTRUCTION_AUGMENTATION_SYSTEM_MESSAGE}
        ]

        if is_search:

            search_messages = [
                {"role": "system", "content": SEARCH_ASSISTANT_SYSTEM_MESSAGE}
            ]
            image.save(".tmp/search_image.png")
            image_data = ImageData(image, image_url=f"https://i.imgur.com/abcd123.jpg/.tmp/search_image.png", local_path=f".tmp/search_image.png")
            
            image_search_result = google_lens_search(image_data)

            search_messages += [
                {
                    "role": "user", "content": [
                        {
                            "type": "image_url",
                            'image_url': {
                                'url': f"data:image/jpeg;base64,{encode_pil_to_base64(image)}"
                            }
                        },
                        {
                            'type': 'text', 
                            'text': f"Here is the image search result:\n{image_search_result}, now please output the no more than 5 most informative phrases (e.g, the combination of event, location, person, action, etc.) need to be further on google search. Directly output the phrases without any other words, each phrase separate by comma."
                        },
                    ]
                }
            ]

            search_keywords = mllm_client.chat_completion(search_messages, timeout=timeout)
            print(f"search_keywords: {search_keywords}")
            # keywords_search_result = []
            # for keyword in search_keywords.split(","):
            #     keywords_search_result.append(google_search(keyword))
            # keywords_search_result = "\n".join(keywords_search_result)
            keywords_search_result = google_search(search_keywords)


            messages += [
                {
                    "role": "user", "content": [
                        {
                            "type": "text", "text": f"Here is the similar image title search result returned by google lens: {image_search_result}\n\nHere is the search result for the keywords in the similar image titles:\n{keywords_search_result}.\n\n Now please summarize the search result in a single paragraph."
                        }
                    ]
                }
            ]

            search_information_summary = mllm_client.chat_completion(search_messages, timeout=timeout)

            print(f"search_information_summary: {search_information_summary}")
            
            messages += [
                {
                    "role": "assistant", "content": [{
                            "type": "text", "text": search_information_summary
                        }]
                },
                {
                    "role": "user", "content": [
                        {
                            "type": "text", "text": f"Now please generate a professional instruction based on user instruction and the above search information. Directly output the instruction without any other words."
                        }
                    ]
                }
            ]

        else:
        
            messages += [   
                {
                    "role": "user", "content": [
                        {
                            "type": "image_url",
                            'image_url': {
                                'url': f"data:image/jpeg;base64,{encode_pil_to_base64(image)}"
                            }
                        },
                        {
                            'type': 'text', 
                            'text': f"User instruction: {query}. \nPlease generate a professional instruction based on user instruction. Directly output the instruction without any other words."
                        }
                    ]
                }
            ]

        return mllm_client.chat_completion(messages, timeout=timeout)


if _name_ == "_main_":
    ia = InstructionAugmenter()
    # print(ia.generate_complex_instruction(Image.open("assets/figs/cybercab.png").convert("RGB"), "Please describe the image.", is_search=True))
    # print(ia.generate_complex_instruction(Image.open("assets/figs/charles_on_the_throne.png").convert("RGB"), "Please describe the image.", is_search=True))
    print(ia.generate_complex_instruction(Image.open("assets/figs/trump_assassination.png").convert("RGB"), "Please describe the image.", is_search=True)) '''