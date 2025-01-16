# Используемые модели и датасеты

## Модели

1. **Llama 3.2 1B Instruct RU**  
   Дообученная на русскоязычном датасете.  
   [Ссылка на модель](https://huggingface.co/Vikhrmodels/Vikhr-Llama-3.2-1B-Instruct)

2. **Llama 3.1 8B Instruct RU**  
   Дообученная на русскоязычном датасете.  
   [Ссылка на модель](https://huggingface.co/Vikhrmodels/Vikhr-Llama3.1-8B-Instruct-R-21-09-24)

---

## Датасеты

1. **forum**  
   Русскоязычный датасет, полученный самостоятельным парсингом психологических форумов (~2000 строк).  
   Формат: `вопрос1 - ответ1`, `вопрос1 - ответ2`, `вопрос2 - ответ1` и т.д.  
   Обработка:
   - Удалены вопросы и ответы с низкой репутацией или короткой длиной.
   - Удалены специфичные для форумов приписки к пользователям, например:  
     `Елена ответил(а) Александру:`, `Елена ответил(а) через 5 минут:`.

2. **forum-gpt**  
   Объединение датасета `forum` с самостоятельно сгенерированными вопросами и ответами GPT-4o психологического характера (~2500–3000 строк).  

3. **STCD**  
   Датасет с Kaggle: [Synthetic Therapy Conversations Dataset](https://www.kaggle.com/datasets/thedevastator/synthetic-therapy-conversations-dataset).  
   Особенности:
   - Содержит диалоги человека с ChatGPT с фокусом на улучшение психологического состояния.  
   - Размер (~8000 строк).
   - Обработка:
     - Удалены имена.
     - Перевод из json в текстовый формат с добавлением тегов <USER> и <ASSISTANT>
     - Переведен на русский язык с использованием `googletrans` (около 8 часов работы скрипта).  

4. **STCD-big-ver1**  
   Расширенная версия STCD (>22,000 строк).  

5. **STCD-GMPM**  
   Объединение:
   - ~11,000 строк из `STCD-big-ver1`.
   - ~5000 строк из [Grand Master Pro Max](https://huggingface.co/datasets/Vikhrmodels/GrandMaster-PRO-MAX), инструктивного датасета, сгенерированного LLM. Выбирал именно те строки, у которых менее 1024 токенов в вопросах и ответах, и в которых встречаются ключевые слова "психология", "учеба", "университет", "стресс" и тд.

---

## Структура проекта

- **test_base_models**  
  Ответы моделей до обучения (finetune).

- **datasets_preview**  
  Первые 25 случайных строк из используемых датасетов.  
  _Полные версии датасетов не размещены из-за ограничений GitHub на размер файлов._

- **Папки моделей после обучения**  
  Пример: `llama-3.2-1b-ep5-STCD`  
  Финальная версия модели после обучения (finetune) на датасете STCD за 5 эпох.  
  Содержимое папки:
  - `train_code.py` — код запуска обучения.
  - `.sh` скрипты — для запуска на кластере.
  - `total_statistics.png` — результаты обучения (графики train loss, eval loss и т.д.).
  - `responses.txt` — ответы модели на вопросы после обучения.

---