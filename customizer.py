from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    pipeline,
)
from datasets import load_dataset, load_from_disk
import os


class Customizer:
    def __init__(self, model_name, output_name):
        self.model_name = model_name
        self.output_name = output_name
        # Initialize the tokenizer and model
        # model_name = "facebook/bart-large-cnn"
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, token=os.getenv("HF_API_KEY")
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, token=os.getenv("HF_API_KEY")
        )
        # Initialize with correct padding configuration
        self.data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            pad_to_multiple_of=8,  # Optimizes for GPU memory alignment
            padding="longest",  # Pads to longest sequence in batch
        )
        self.tokenized_datasets = None

    # Preprocess the dataset
    def preprocess_function(self, examples):
        inputs = ["summarize: " + doc for doc in examples["article"]]
        # Input processing
        model_inputs = self.tokenizer(
            inputs,
            max_length=1024,
            truncation=True,
            padding=False,  # Let data collator handle padding
        )

        # Label processing
        labels = self.tokenizer(
            text_target=examples["highlights"],
            max_length=128,
            truncation=True,
            padding=False,  # Critical for dynamic padding
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def tokenize_dataset(self):
        self.tokenized_datasets = self.dataset.map(
            self.preprocess_function,
            batched=True,
            load_from_cache_file=True,
            cache_file_names={
                "train": "./cache/train_cache.arrow",
                "validation": "./cache/val_cache.arrow",
                "test": "./cache/test_cache.arrow",
            },
        )
        self.tokenized_datasets.save_to_disk("./cnn_token_cache")
        print("Saved tokens to disk")

    def load_dataset(self):
        # Load the CNN/DailyMail dataset
        self.dataset = load_dataset(
            "cnn_dailymail",
            "3.0.0",
            cache_dir="./hf_data_cache",
            token=os.getenv("HF_API_KEY"),
        )

        try:
            self.tokenized_datasets = load_from_disk("./cnn_token_cache")
        except:
            print("nothing on the disk, processing fresh")

        if (
            self.tokenized_datasets == None
            or self.tokenized_datasets.num_rows == 0
        ):
            print("i think there was nothing in the cache")
            self.tokenize_dataset()
        else:
            print("i think I loaded maps from the cache!")

    def train_and_save(self):
        # Set up training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir="./results",
            eval_strategy="epoch",
            learning_rate=5e-4,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=1,
            # padding='longest',  # Explicit padding strategy
            predict_with_generate=True,
        )

        # Initialize the trainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_datasets["train"],
            eval_dataset=self.tokenized_datasets["validation"],
            processing_class=self.tokenizer,
            data_collator=self.data_collator,
        )

        # Start training
        trainer.train()
        # Save the model
        self.model.save_pretrained(f"./{self.output_name}")
        self.tokenizer.save_pretrained(f"./{self.output_name}")
        return self.output_name

        # self.model.save_pretrained(f"./news_summarizer_model")
        # self.tokenizer.save_pretrained(f"./news_summarizer_model")
