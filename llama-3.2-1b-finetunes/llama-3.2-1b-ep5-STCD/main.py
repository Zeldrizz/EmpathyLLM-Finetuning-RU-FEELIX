import os
import torch
import sys

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
from peft import (
    LoraConfig,
    get_peft_model
)
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig, setup_chat_format

if not torch.cuda.is_available():
    print("GPU недоступно")
    sys.exit(1)

base_model = "/home/mrtaktarov/FEELIX_bot/models/Llama-3.2-1B-Instruct-RU"
new_model = "llama-3.2-1b-it-Emotional_Psychology-Support-ChatBot-RU-STCD"
dataset_name = "/home/mrtaktarov/FEELIX_bot/datasets/STCD_RU.csv"

instruction = """
Ты — эмпатичный и доброжелательный собеседник.
Твоя задача — помогать людям справляться с эмоциональными трудностями и улучшать их настроение, особенно студентам университета.
Фокусируйся на собеседнике и его чувствах, избегая обсуждения себя или своей роли.
Используй психологические методики, чтобы задавать вопросы, давать поддержку и предлагать полезные мысли.
Всегда оставайся добрым и терпеливым.
При возникновении конфликтных тем переводи разговор в другое русло.
"""

torch_dtype = torch.float16

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto",
    torch_dtype=torch_dtype,
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

dataset = load_dataset("csv", data_files=dataset_name)["train"]
train_test_split = dataset.train_test_split(test_size=0.2, shuffle=True)
train_dataset = train_test_split["train"]
test_dataset = train_test_split["test"]

def format_chat_template(row):
    row_json = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": row["request"]},
        {"role": "assistant", "content": row["response"]},
    ]
    row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
    return row

train_dataset = train_dataset.map(format_chat_template, num_proc=1)
test_dataset = test_dataset.map(format_chat_template, num_proc=1)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"]
)

tokenizer.chat_template = None
model, tokenizer = setup_chat_format(model, tokenizer)
model = get_peft_model(model, peft_config)

sft_config = SFTConfig(
    output_dir=new_model,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=2,
    optim="adamw_hf",
    num_train_epochs=5,
    do_eval=True,
    evaluation_strategy="steps",
    eval_steps=500,
    logging_steps=1,
    warmup_steps=10,
    logging_strategy="steps",
    learning_rate=2e-4,
    fp16=False,
    bf16=False,
    group_by_length=True,
    max_seq_length=512,
    dataset_text_field="text",
    packing=False,
    save_strategy="steps",
    save_steps=2000,
    report_to=None
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    peft_config=peft_config,
    processing_class=tokenizer,
    args=sft_config
)

trainer.train()
trainer.save_model(new_model)