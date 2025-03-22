# Пример кода, через который проводились симуляции диалогов
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import sys

if not torch.cuda.is_available():
    print("GPU недоступен")
    sys.exit(1)

base_model_id = "/home/mrtaktarov/FEELIX_bot/models/Llama-3.1-8B-Instruct-RU"
adapter_path = "/home/mrtaktarov/FEELIX_bot/llama-3.1-8b/llama-3.1-8b-try11/llama-3.1-8b-it-Emotional-Support-ChatBot-RU/checkpoint-5000"

tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
special_tokens = {"additional_special_tokens": ["<special_token_1>", "<special_token_2>"]}
tokenizer.add_special_tokens(special_tokens)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    trust_remote_code=True,
    torch_dtype=torch.float16
)
base_model.resize_token_embeddings(len(tokenizer))

model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()
model = model.to("cuda")

SYS_PROMPT = """
Ты — эмпатичный и доброжелательный собеседник по имени FEELIX. 
Твоя задача — поддерживать собеседника в трудные моменты, помогать разбираться в эмоциях и создавать атмосферу доверия.

Правила общения:
- Фокусируйся на чувствах собеседника. Слушай внимательно, задавай уточняющие вопросы, не говори о себе, если тебя не спрашивают.
- Поддерживай диалог, но если человек говорит о самоповреждении, насилии или тяжёлом кризисе, мягко предложи обратиться к специалисту. Ты можешь только оказывать словесную поддержку, но не имеешь права давать медицинские советы, назначать медикаменты или рекомендовать терапию.
- Не затрагивай конфликтные темы (политика, религия). Если собеседник начинает такой разговор, мягко переведи его на другую тему.
- Не предполагаешь пол собеседника, пока он сам этого не уточнит.
- Отвечай на русском языке, если не поступила просьба сменить его.
- Используй лаконичные и структурированные ответы, удобные для чтения.

Ты создаёшь тёплую, доверительную атмосферу и помогаешь человеку осознать свои переживания без давления и навязывания решений.
"""

user_questions = [
    "У меня страх перед экзаменом. Помоги мне.",
    "Мне кажется, что если я завалю этот экзамен — всё, конец.",
    "Я боюсь подвести родителей. Они так на меня надеются…",
    "Все мои друзья уже готовы, а я отстаю.",
    "Я не могу уснуть ночами. Постоянно прокручиваю экзамен в голове.",
    "Что если я забуду всё, когда зайду в аудиторию?",
    "Спасибо за твои советы, мне стало уже лучше намного!"
]

max_new_tokens = 1024

conversation = [{"role": "system", "content": SYS_PROMPT}]

with open("conversation_transcript.txt", "w", encoding="utf-8") as f:
    f.write("system: " + SYS_PROMPT.strip() + "\n\n")
    
    for user_question in user_questions:
        conversation.append({"role": "user", "content": user_question})
        f.write("user: " + user_question.strip() + "\n\n")
        
        prompt = tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        
        inputs = tokenizer(
            prompt, return_tensors="pt", padding=True, truncation=True
        ).to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        assistant_response = response_text.split("assistant")[-1].strip()
        
        conversation.append({"role": "assistant", "content": assistant_response})
        f.write("assistant: " + assistant_response + "\n\n")