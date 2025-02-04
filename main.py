from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# from datasets import load_dataset
from dotenv import load_dotenv
from customizer import Customizer
from scorer import Scorer
from summary_generator import SummaryGenerator as sg
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
    generator = sg()
    basline_dict = generator.generate_summaries("facebook/bart-large-cnn")
    comparison_dict_1 = generator.generate_summaries("google/pegasus-xsum")
    comparison_dict_2 = generator.generate_summaries("t5-small")

    scoring_agent = Scorer(
        basline_dict,
        {"google/pegasus-xsum": comparison_dict_1, "t5-small": comparison_dict_2},
    )
    scoring_agent.compute_comparison_tensors()


if __name__ == "__main__":
    main()
