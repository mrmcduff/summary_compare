from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    pipeline,
)
from sentence_transformers import SentenceTransformer, util
from typing import Dict, List
import numpy as np
import os


class Scorer:
    def __init__(
        self,
        indexed_base_sentences: Dict[str, str],
        indexed_test_sentences: Dict[str, Dict[str, str]],
    ):
        self.embedding_model = SentenceTransformer(
            "all-MiniLM-L6-v2", token=os.getenv("HF_API_KEY")
        )
        self.indexed_base_sentences = indexed_base_sentences
        self.indexed_test_sentences = indexed_test_sentences
        self.comparison_tensors = {}
        self.model_scores: Dict[str, List[float]] = {}
        self.compute_baseline()
        self.compute_comparison_tensors()
        self.compute_similarity_lists()

    def compute_baseline(self):
        key_list = []
        sentence_list = []
        for key, value in self.indexed_base_sentences.items():
            key_list.append(key)
            sentence_list.append(value)
        embeddings_list = self.embedding_model.encode(
            sentences=sentence_list, convert_to_numpy=True, normalize_embeddings=True
        )
        self.key_list = (
            key_list  # ensure we loop over the same sentences in the same order
        )
        self.baseline_tensor = embeddings_list

    def compute_comparison_tensors(self):
        model_list = []
        total_sentences = {}
        for model_name, model_dict in self.indexed_test_sentences.items():
            model_list.append(model_name)
            sentence_list = []
            for key in self.key_list:
                sentence_list.append(model_dict.get(key))

            total_sentences[model_name] = sentence_list
            embeddings_list = self.embedding_model.encode(
                sentences=sentence_list,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            self.comparison_tensors[model_name] = embeddings_list

    def compute_similarity_lists(self):
        self.cosine_values = {}

        for model_name, tensor in self.comparison_tensors.items():
            self.model_scores[model_name] = []
            for idx, row in enumerate(self.baseline_tensor):
                man_cos_sim = self.manual_cosine_sim(row, tensor[idx])
                self.model_scores[model_name].append(man_cos_sim)
                print(
                    f"In row {idx} of model {model_name}, found value of {man_cos_sim}"
                )
        print(self.model_scores)

    def manual_cosine_sim(self, vec_a: List[float], vec_b: List[float]) -> float:
        return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
