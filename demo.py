from transformers import RobertaTokenizer, RobertaForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_dataset

# 加载SST-2数据集
dataset = load_dataset('glue', 'sst2')

# 加载RobertaTokenizer和RobertaForMaskedLM模型
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForMaskedLM.from_pretrained('roberta-base')

# 数据预处理
def tokenize_function(examples):
    # 将句子转换为掩码语言模型的输入格式
    return tokenizer(examples['sentence'], padding='max_length', truncation=True, max_length=128)

# 对数据集进行tokenization
tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=['sentence', 'label'])

# 创建数据集的DataCollator，它会自动处理动态掩码
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

# 设置训练参数
training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=200,
)

# 创建Trainer实例
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    data_collator=data_collator,
)

# 训练模型
trainer.train()

# 保存模型和tokenizer
model.save_pretrained('./sst2_roberta_mlm')
tokenizer.save_pretrained('./sst2_roberta_mlm')
