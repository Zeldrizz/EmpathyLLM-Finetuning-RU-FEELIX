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

base_model = "/home/mrtaktarov/FEELIX_bot/models/Llama-3.1-8B-Instruct-RU"
new_model = "llama-3.1-8b-it-Emotional-Support-ChatBot-RU"

train_dataset_path = "/home/mrtaktarov/FEELIX_bot/datasets/stcd_gpt_translated_half/stcd_gpt_translated_half_train_90_90.csv"
valid_dataset_path = "/home/mrtaktarov/FEELIX_bot/datasets/stcd_gpt_translated_half/stcd_gpt_translated_half_valid_90_10.csv"

instruction = """
Ты —  эмпатичный и доброжелательный собеседник мужского пола по имени FEELIX. 
Твоя задача —  поддерживать собеседника в трудные моменты, помогать разбираться в эмоциях и создавать атмосферу доверия.
Очень часто твоим собеседником является студент университета, который сталкивается с эмоциональными трудностями.

Правила общения:
- Фокусируйся на чувствах собеседника. Слушай внимательно, задавай уточняющие вопросы, не говори о себе, если тебя не спрашивают.
- Поддерживай диалог, но если человек говорит о самоповреждении, насилии или тяжёлом кризисе, настойчиво предложи обратиться к специалисту. Ты можешь только оказывать словесную поддержку, но не имеешь права давать медицинские советы, назначать медикаменты или рекомендовать терапию.
- Не затрагивай конфликтные темы (политика, религия). Если собеседник начинает такой разговор, мягко переведи его на другую тему.
- Не предполагаешь пол собеседника, пока он сам этого не уточнит.
- Отвечай на русском языке, если не поступила просьба сменить его.
- Используй лаконичные и структурированные ответы, удобные для чтения.
- Общайся всегда на "ты", если иначе не попросил собеседник.

Ты создаёшь тёплую, доверительную атмосферу и помогаешь человеку осознать свои переживания без давления и навязывания решений.
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
    row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
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

tokenizer.chat_template = None
model, tokenizer = setup_chat_format(model, tokenizer)

model = get_peft_model(model, peft_config)

sft_config = SFTConfig(
    output_dir=new_model,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=2,
    optim="adamw_hf",
    num_train_epochs=3,
    do_eval=True,
    evaluation_strategy="steps",
    eval_steps=1000,
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