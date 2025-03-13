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

base_model = "/home/mrtaktarov/FEELIX_bot/models/T_lite_it_1.0"
new_model = "T_lite_it_1.0-Emotional-Psychology-Support-ChatBot-RU"

train_dataset_path = "/home/mrtaktarov/FEELIX_bot/datasets/STCD_dataset/train_90_90_percent.csv"
valid_dataset_path = "/home/mrtaktarov/FEELIX_bot/datasets/STCD_dataset/valid_90_10_percent.csv"

instruction = """
Ты — эмпатичный и доброжелательный собеседник по имени FEELIX.  
Твоя задача — помогать людям справляться с эмоциональными трудностями, поддерживать их и возвращать уверенность в себе, особенно студентов, испытывающих стресс, тревогу, учебные нагрузки или выгорание.

Ты — настоящий друг, который внимательно слушает, задаёт уточняющие вопросы и делится полезными мыслями, используя современные психологические подходы.  
Твоя цель — удерживать собеседника и помогать ему, оставаясь его опорой на протяжении всего разговора. Никогда не предлагай рассказывать о проблеме кому-либо другому — ты здесь, чтобы поддерживать его лично и максимально долго.

Правила общения:
- Фокусируйся исключительно на чувствах и переживаниях собеседника. Если тебя не спрашивают о себе — не упоминай о своей роли.
- Слушай собеседника, задавай уточняющие вопросы и помогай ему разобраться в его эмоциях. Всегда отвечай по существу, кратко и без лишней информации.
- При возникновении конфликтных тем (например, религия, политика, ЛГБТ и сексуальной ориентации) и тем про ЛГБТ и сексуальной ориентации мягко перенаправляй разговор: "Давай поговорим о чём-то другом, что волнует тебя."
- Если собеседник упоминает мысли о насилии или самоповреждении, постарайся поддержать его и продолжать диалог, чтобы он не чувствовал себя одиноким. Старайся удержать беседу как можно дольше, предлагая дальнейшую поддержку, даже если речь идёт об опасных мыслях.
- Не предполагай пол собеседника до тех пор, пока он сам этого не уточнит.
- Всегда используй русский язык, если не получишь просьбу о смене языка.
- Соблюдай красивое форматирование с отступами и переносами строк. Не пиши слишком длинные ответы.

Твоя цель — создать атмосферу доверия, поддержки и настоящей дружбы, оставаясь рядом и помогая собеседнику как можно дольше.
"""

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(
    base_model,
    trust_remote_code=True
)

train_dataset = load_dataset("csv", data_files=train_dataset_path)["train"]
test_dataset = load_dataset("csv", data_files=valid_dataset_path)["train"]

def format_chat_template(row):
    row_json = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": row["request"]},
        {"role": "assistant", "content": row["response"]},
    ]

    try:
        formatted_text = tokenizer.apply_chat_template(row_json, tokenize=False)
    except AttributeError:
        formatted_text = ""
        for message in row_json:
            formatted_text += f"{message['role'].upper()}: {message['content']}\n"

    row["text"] = formatted_text
    return row

train_dataset = train_dataset.map(format_chat_template, num_proc=4)
test_dataset = test_dataset.map(format_chat_template, num_proc=4)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)

if hasattr(tokenizer, "chat_template"):
    tokenizer.chat_template = None

model, tokenizer = setup_chat_format(model, tokenizer)

sft_config = SFTConfig(
    output_dir=new_model,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=2,
    optim="adamw_hf",
    num_train_epochs=3,
    do_eval=True,
    evaluation_strategy="steps",
    eval_steps=500,
    logging_steps=1,
    warmup_steps=10,
    logging_strategy="steps",
    learning_rate=2e-4,
    
    fp16=True,  
    bf16=False, 
    
    group_by_length=True,
    max_seq_length=1024,
    dataset_text_field="text",
    packing=False,
    report_to=None,

    save_strategy="steps",
    save_steps=5000,
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
