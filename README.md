# summerization
#it summerizes the passage

#importing the transformers model
from transformers import AutoTokenizer, BartForConditionalGeneration

#downloading the model
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

#taking the input text
ARTICLE_TO_SUMMARIZE = input(print("Enter the text: \n"))
inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors="pt")

#geneerating the summery
summary_ids = model.generate(inputs["input_ids"], num_beams=2, min_length=0, max_length=20)
tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
