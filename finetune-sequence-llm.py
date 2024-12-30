from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoModelForMaskedLM
from datasets import load_dataset, load_metric
from peft import LoraConfig, get_peft_model, PeftConfig, PeftModel, AutoPeftModelForSequenceClassification
import torch
import argparse
import numpy as np

def train(args):

    model_path = args.model_name
    if args.model_name == "FacebookAI/roberta-base":
        output_dir = f'roberta_base_output'
    max_seq_length = 256

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        # num_labels=2,  # SST-2 has two labels: positive and negative
        device_map="auto"
    )

    # EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN

    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # model.resize_token_embeddings(len(tokenizer))
    # model.config.pad_token_id = tokenizer.pad_token_id
  
    dataset = load_dataset("glue", "sst2")["train"]

    def tokenize_function(examples):
        inputs = examples["sentence"]
        for idx, value in enumerate(examples["sentence"]):
            examples["sentence"][idx] = f"{value}. It was "
        return tokenizer(examples["sentence"], padding="max_length", truncation=True, max_length=max_seq_length)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Remove unnecessary columns and set the format for PyTorch tensors
    tokenized_dataset = tokenized_dataset.remove_columns(["sentence", "idx"])
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format("torch")

    training_args = TrainingArguments(
        per_device_train_batch_size=16,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=1,
        learning_rate=2e-8,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_hf",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=output_dir,
        save_steps=1000,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(output_dir)

def evaluate(args):

    model_path = f'roberta_base__output/checkpoint-1000'  # Use the path where the model is saved
    max_seq_length = 256

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    validation_dataset = load_dataset("glue", "sst2")["validation"]

    def tokenize_function(examples):
        inputs = examples["sentence"]
        for idx, value in enumerate(examples["sentence"]):
            examples["sentence"][idx] = f"{value}. It was "
        return tokenizer(examples["sentence"], padding="max_length", truncation=True, max_length=max_seq_length)

    tokenized_validation_dataset = validation_dataset.map(tokenize_function, batched=True)
    tokenized_validation_dataset = tokenized_validation_dataset.remove_columns(["sentence", "idx"])
    tokenized_validation_dataset = tokenized_validation_dataset.rename_column("label", "labels")
    tokenized_validation_dataset.set_format("torch")

    # Load the accuracy metric
    metric = load_metric("accuracy")

    # Define the compute_metrics function to calculate accuracy
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,  # Pass the compute_metrics function here
    )

    # Evaluate the model and print accuracy
    results = trainer.evaluate(eval_dataset=tokenized_validation_dataset)
    print("Accuracy:", results["eval_accuracy"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2-1.5B', choices=['Qwen/Qwen2-1.5B','google/gemma-2-2b','FacebookAI/roberta-base'])

    args = parser.parse_args()
    train(args)
    evaluate(args)
    # train()
