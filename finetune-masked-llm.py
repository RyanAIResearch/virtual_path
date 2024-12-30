from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoModelForMaskedLM
from datasets import load_dataset, load_metric
from fastNLP import DataSet, Instance
from fastNLP.io import Loader, DataBundle
from peft import LoraConfig, get_peft_model, PeftConfig, PeftModel, AutoPeftModelForSequenceClassification
from functools import partial
import torch
import argparse
import numpy as np

def process_dataset(dataset):
    # 过滤掉长度超过512的样本，并保留原始的 datasets.Dataset
    def filter_function(example):
        return len(example["input_ids"]) <= 512
    
    dataset = dataset.filter(filter_function)

    return dataset

def train(args):

    model_path = args.model_name
    if args.model_name == "FacebookAI/roberta-base":
        output_dir = f'roberta_base_output'
    max_seq_length = 256

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForMaskedLM.from_pretrained(
        model_path,
        device_map="auto"
    )
  
    dataset = load_dataset("glue", "sst2")["train"]

    def convert_to_features(example_batch, tokenizer):
        input_encodings = tokenizer.batch_encode_plus(example_batch['input_text'])
        target_encodings = tokenizer.batch_encode_plus(example_batch['target_text'], add_special_tokens=False)
        mask_pos = []
        for input_ids in input_encodings['input_ids']:
            mask_pos.append(input_ids.index(tokenizer.mask_token_id))
        encodings = {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'mask_pos': mask_pos,
            'labels': target_encodings['input_ids'],
        }
        return encodings


    def tokenize_function(examples):
        inputs = examples['sentence']
        examples['input_text'] = inputs
        target_text = []

        for idx, value in enumerate(inputs):
            masked_input = f"{value} . It was {tokenizer.mask_token} ."
            examples['input_text'][idx] = masked_input

            if examples['label'][idx] == 0:
                target_text.append("bad")
            else:
                target_text.append("great")

        examples['target_text'] = target_text
        return examples

    tokenized_dataset = dataset.map(tokenize_function, batched=True, batch_size=8)

    tokenized_dataset = tokenized_dataset.map(partial(convert_to_features, tokenizer=tokenizer), batched=True, load_from_cache_file=False, batch_size=8)
    # dataset = dataset.remove_columns(["input", "output"])
    tokenized_dataset = tokenized_dataset.remove_columns(["sentence", "label", "input_text", "target_text"])

    train_dataset = process_dataset(tokenized_dataset)

    # def format_dataset(examples):
    #     # 先将 input_ids 和 attention_mask 转换为 Tensor
    #     input_ids = torch.tensor(examples['input_ids'], dtype=torch.long)
    #     attention_mask = torch.tensor(examples['attention_mask'], dtype=torch.long)

    #     # 创建 labels 张量，初始化为 -100，形状与 input_ids 一致
    #     labels = torch.full_like(input_ids, -100)

    #     mask_pos = examples['mask_pos']

    #     # 为每个样本在 mask_pos 位置设置正确的标签
    #     for i, pos in enumerate(mask_pos):
    #         labels[i, pos] = examples['labels'][i]

    #     # 返回带有 input_ids、attention_mask 和 labels 的字典
    #     return {
    #         'input_ids': input_ids,
    #         'attention_mask': attention_mask,
    #         'labels': labels
    #     }

    print("train_dataset: ", train_dataset)
    # train_dataset = train_dataset.map(format_dataset, batched=True, batch_size=8)
    # Remove unnecessary columns and set the format for PyTorch tensors
    # tokenized_dataset = tokenized_dataset.remove_columns(["sentence", "idx"])
    # tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    # tokenized_dataset.set_format("torch")


    training_args = TrainingArguments(
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
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
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(output_dir)

def evaluate(args):

    model_path = f'roberta_base__output/checkpoint-1000'  # Use the path where the model is saved
    max_seq_length = 256

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    base_model = AutoModelForSequenceClassification.from_pretrained(model_path)

    model = PeftModel.from_pretrained(base_model, model_path)

    # model.config.pad_token_id = tokenizer.pad_token_id
    # peft_config = PeftConfig.from_pretrained(model_path)
    # model = PeftModel.from_pretrained(base_model, peft_config)

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
