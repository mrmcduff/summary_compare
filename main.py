from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# from datasets import load_dataset
from dotenv import load_dotenv
from customizer import Customizer
from scorer import Scorer
# import os

load_dotenv()


def main():
    # Initialize the tokenizer and model
    # model_name = "facebook/bart-large-cnn"
    # customizer = Customizer("facebook/bart-large-cnn", "first_try")
    # customizer.load_dataset()
    # name_out = customizer.train_and_save()
    # print(f"name created is {name_out}")

    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    sample_dict = {
        "a": "I am fit as a fiddle",
        "b": "It was freezing outside today",
        "c": "Weather recently was on the cold side",
        "d": "The Lakers made a blockbuster trade just before the deadline",
        "e": "One is the loneliest number",
    }
    sample_dict_2 = {
        "a": "I am in good health",
        "b": "It froze today",
        "c": "Recent weather was cold",
        "d": "The lake busted blocks on a deadline",
        "e": "A person is lonely",
    }
    sample_dict_3 = {
        "a": "I fit fiddles",
        "b": "It was cold",
        "c": "Weather recently was beside itself",
        "d": "The Lakers traded blocks at the deadline",
        "e": "One is lonely",
    }
    scoring_agent = Scorer(
        sample_dict, {"first_model": sample_dict_2, "second_model": sample_dict_3}
    )
    scoring_agent.compute_comparison_tensors()


if __name__ == "__main__":
    main()
