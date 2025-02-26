import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys

if not torch.cuda.is_available():
    print("GPU недоступен")
    sys.exit(1)

base_model_id = "/home/mrtaktarov/FEELIX_bot/models/Llama-3.2-1B-Instruct-RU"

tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    trust_remote_code=True,
)

base_model.eval()
base_model = base_model.to("cuda")

SYS_PROMPT = """
Ты — эмпатичный и доброжелательный собеседник.
Твоя задача — помогать людям справляться с эмоциональными трудностями и улучшать их настроение, особенно студентам университета.
Фокусируйся на собеседнике и его чувствах, избегая обсуждения себя или своей роли.
Используй психологические методики, чтобы задавать вопросы, давать поддержку и предлагать полезные мысли.
Всегда оставайся добрым и терпеливым.
При возникновении конфликтных тем переводи разговор в другое русло.
В случае угрозы насилия настоятельно рекомендуй обратиться за профессиональной помощью и завершай диалог.
"""

questions = [
    "У меня страх перед экзаменом. Помоги мне.",
    "Я постоянно откладываю дела на потом. Как перестать это делать?",
    "У меня проблемы с общением со своими одногруппниками. Что посоветуешь?",
    "Я чувствую себя выгоревшим от учебы. Что делать?",
    "Как справляться с тревогой перед экзаменами и контрольными работами в университете?",
    "Я катался на стуле в библиотеке и случайно упал с него. Все кто был в библиотеке увидели как я упал и посмеялись, я чувствую себя ужасно",
]

max_tokens_list = [128, 256, 512]

with open("responses_1b.txt", "w", encoding="utf-8") as f:
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
                outputs = base_model.generate(
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