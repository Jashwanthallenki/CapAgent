from PIL import Image

from capagent.tools import visual_question_answering_image, change_caption_sentiment, count_words, shorten_caption

# Get base description with VQA
base_description = visual_question_answering_image(
    "Describe the kitten's expression and overall scene in detail",
    image_1,
    show_result=False
)

# Transform to humorous tone while adding reaction phrases
humorous_caption = change_caption_sentiment(
    caption=f"{base_description} Include phrases like 'What just happened?' and 'Did you see that?'",
    sentiment="humorous",
    show_result=False
)

# Ensure concise length (under 20 words)
word_count = count_words(humorous_caption, show_result=False)
if word_count > 20:
    final_caption = shorten_caption(humorous_caption, max_words=20, show_result=False)
else:
    final_caption = humorous_caption

# Display results
print(f"Final caption ({count_words(final_caption, show_result=False)} words): {final_caption}")
