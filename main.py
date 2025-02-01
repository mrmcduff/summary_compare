from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# from datasets import load_dataset
from dotenv import load_dotenv
from customizer import Customizer
# import os

load_dotenv()


def main():
    # Initialize the tokenizer and model
    # model_name = "facebook/bart-large-cnn"
    customizer = Customizer("facebook/bart-large-cnn", "first_try")
    customizer.load_dataset()
    # name_out = customizer.train_and_save()
    # print(f"name created is {name_out}")

    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


if __name__ == "__main__":
    main()
