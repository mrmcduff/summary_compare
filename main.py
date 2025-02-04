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
    generator.set_sample_articles()
    basline_dict = generator.generate_summaries("facebook/bart-large-cnn")
    print("baseline")
    print(basline_dict)
    comparison_dict_1 = generator.generate_summaries("google/pegasus-xsum")
    print("xsum dictionary")
    print(comparison_dict_1)
    comparison_dict_2 = generator.generate_summaries("t5-small")
    print("t5-small dictionary")
    print(comparison_dict_2)


    print("Finished generating samples")
    scoring_agent = Scorer(
        basline_dict,
        {"google/pegasus-xsum": comparison_dict_1, "t5-small": comparison_dict_2},
    )
    print("Preparing scores")
    scoring_agent.compute_similarity_lists()


if __name__ == "__main__":
    main()
