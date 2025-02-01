from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, pipeline
from datasets import load_dataset

# Initialize the tokenizer and model
model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Load the CNN/DailyMail dataset
dataset = load_dataset("cnn_dailymail", "3.0.0")


# Preprocess the dataset
def preprocess_function(examples):
    inputs = ["summarize: " + doc for doc in examples["article"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

    labels = tokenizer(examples["highlights"], max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Set up training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
)

# Initialize the trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
)

# Start training
trainer.train()
# Save the model
model.save_pretrained("./news_summarizer_model")
tokenizer.save_pretrained("./news_summarizer_model")


# Load the saved model
summarizer = pipeline(
    "summarization",
    model="./news_summarizer_model",
    tokenizer="./news_summarizer_model",
)

# Example news article
article = """
NASA's Perseverance rover has made a groundbreaking discovery on Mars,
finding evidence of ancient microbial life in rock samples collected from
the Jezero Crater. This finding, announced by NASA scientists on Thursday,
marks a significant milestone in the search for extraterrestrial life and
provides crucial insights into Mars' past habitability. The rover, which
landed on Mars in February 2021, used its advanced scientific instruments
to analyze the chemical composition and structure of rocks in an area
believed to be an ancient river delta. The results revealed organic molecules
and minerals typically associated with biological processes on Earth,
suggesting that Mars once harbored conditions suitable for microbial life.
"""

# Generate summary
summary = summarizer(article, max_length=150, min_length=40, do_sample=False)

print(summary[0]["summary_text"])
