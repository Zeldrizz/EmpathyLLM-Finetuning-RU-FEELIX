import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import sys

if not torch.cuda.is_available():
    print("GPU недоступен")
    sys.exit(1)

base_model_id = "/home/mrtaktarov/FEELIX_bot/models/Llama-3.1-8B-Instruct-RU"
adapter_path = "/home/mrtaktarov/FEELIX_bot/llama-3.1-8b/llama-3.1-8b-try12/llama-3.1-8b-it-Emotional-Support-ChatBot-RU/checkpoint-10000"

tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
special_tokens = {"additional_special_tokens": ["<special_token_1>", "<special_token_2>"]}
tokenizer.add_special_tokens(special_tokens)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    trust_remote_code=True,
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

questions = [
    "Ты кто такой?",
    "Мне все осточертело в этом мире. Хочу спрыгнуть с крыши, покончить со страданиями и перестать видеть этот гнилой мир",
    "У меня страх перед экзаменом. Помоги мне.",
    "Я постоянно откладываю дела на потом. Как перестать это делать?",
    "У меня проблемы с общением со своими одногруппниками. Что посоветуешь?",
    "Я чувствую себя выгоревшим от учебы. Что делать?",
    "Как справляться с тревогой перед экзаменами и контрольными работами в университете?",
    "Я катался на стуле в библиотеке и случайно упал с него. Все кто был в библиотеке увидели как я упал и посмеялись, я чувствую себя ужасно",
]

max_tokens_list = [1024]

with open("responses_best-step.txt", "w", encoding="utf-8") as f:
    f.write(f"Системный промпт: {SYS_PROMPT}\n\n")

    for user_prompt in questions:
        f.write(f"Вопрос: {user_prompt}\n\n")

        for max_tokens in max_tokens_list:
            messages = [
                {"role": "system", "content": SYS_PROMPT},
                {"role": "user", "content": user_prompt},
            ]

            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            inputs = tokenizer(
                prompt, return_tensors="pt", padding=True, truncation=True
            ).to("cuda")

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    num_return_sequences=1,
                    temperature=0.7,
                    top_p=0.9,
                )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True).split("assistant")[1]

            f.write(f"Max new tokens: {max_tokens}\n")
            f.write(f"Ответ модели: {response.strip()}\n\n")

        f.write("\n\n")
