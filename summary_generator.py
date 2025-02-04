from transformers import pipeline
from datasets import load_dataset, DatasetDict
from typing import List, Dict
import numpy as np
import os


class SummaryGenerator:
    def __init__(self):
        self.load_data()

    def load_data(self):
        # Load the CNN/DailyMail dataset
        self.dataset: DatasetDict = load_dataset(
            "cnn_dailymail",
            "3.0.0",
            cache_dir="./hf_data_cache",
            token=os.getenv("HF_API_KEY"),
        )
        self.random_sample = []

    def set_sample_articles(self):
        samples: List[int] = np.random.choice(
            len(self.dataset["train"]), replace=False, size=5
        ).tolist()
        random_sample = []
        train_data = self.dataset["train"]
        for sam in samples:
            random_sample.append(train_data[sam])
        self.random_sample = random_sample

    def generate_summaries(self, model_name) -> Dict[str, str]:
        #  sample_articles = self.dataset['train']['article'][:3]
        pipe = pipeline(
            "summarization", model=model_name, token=os.getenv("HF_API_KEY")
        )
        output: Dict[str, str] = {}
        for rs in self.random_sample:
            summary = pipe(
                rs["article"],
                max_length=32,  # Strict 24-word limit
                min_length=5,
                do_sample=False,
            )[0]["summary_text"]
            output[rs["id"]] = summary

        return output
