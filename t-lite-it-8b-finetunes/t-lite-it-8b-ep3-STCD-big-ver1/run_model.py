import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import sys

if not torch.cuda.is_available():
    print("GPU недоступен")
    sys.exit(1)

base_model_id = "/home/mrtaktarov/FEELIX_bot/models/T_lite_it_1.0"
adapter_path = "/home/mrtaktarov/FEELIX_bot/t-lite-it-try3/T_lite_it_1.0-Emotional-Psychology-Support-ChatBot-RU/checkpoint-20000"

tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
special_tokens = {"additional_special_tokens": ["<special_token_1>", "<special_token_2>"]}
tokenizer.add_special_tokens(special_tokens)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto"
).eval()
# base_model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

#model = PeftModel.from_pretrained(base_model, adapter_path)
#model.eval()
#model = model.to("cuda")

model = PeftModel.from_pretrained(
    base_model,
    adapter_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

SYS_PROMPT = """
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

with open("responses.txt", "w", encoding="utf-8") as f:
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
            ).to(model.device)

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
