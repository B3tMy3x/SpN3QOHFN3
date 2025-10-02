Проект: LangGraph pipeline с LLM-critic, Enhance Prompt и Fix Error Prompt

Что сделано
- Реализованы два требуемых потока:
  - llm critic → enhance prompt
  - llm critic → sql executor → fix error prompt
- Используется LangGraph для построения графа состояний.
- Модель: DeepSeek Coder 6.7B Instruct (локально, без API), путь: `kaggle/input/deepseek-coder`.
- Исполнитель SQL реализован как модель (LLM), без подключения к БД: валидирует запрос по схеме и имитирует фидбек об ошибках.

Структура
- `langpipe/llm.py` — локальный клиент на Transformers, грузит модель из `kaggle/input/deepseek-coder`.
- `langpipe/critic.py` — промпты и вызовы LLM для critic/enhance/fix.
- `langpipe/sql.py` — LLM-исполнитель SQL (валидация и имитация результатов/ошибок, без БД).
- `langpipe/graph.py` — сборка графа LangGraph с нужными узлами и ветвлениями.
- `examples/run.py` — пример с двумя ветками (без реальной БД).

Установка
1) (Опционально) создайте виртуальное окружение
   python -m venv .venv
   source .venv/bin/activate

2) Установите зависимости
   pip install -r requirements.txt
   (На Kaggle обычно уже есть torch; если нет — установите совместимую версию.)

Настройка LLM (DeepSeek Coder 6.7B локально)
- Поместите модель в `kaggle/input/deepseek-coder` (или укажите путь переменной `LOCAL_MODEL_DIR`).
- Формат HF (config.json, tokenizer, веса). Обёртка сама найдёт подпапку с `config.json`, если модель лежит внутри датасета.

Пример:
  export LOCAL_MODEL_DIR="kaggle/input/deepseek-coder"

Запуск примера
  python examples/run.py

Как это работает
- Точка входа графа — узел `llm_critic`:
  - LLM выдает JSON с полем `route`: `enhance_prompt` или `execute_sql`.
- Ветка `enhance_prompt`:
  - Узел формирует улучшенный промпт для будущего генератора SQL.
- Ветка `execute_sql`:
  - Исполнитель — модель (без БД): валидирует SQL по схеме и возвращает success/error.
  - При ошибке → `fix_error_prompt` (LLM). На этом этапе цикл завершается; внешний генератор возьмет `fixed_prompt`.
 - Контекст: во все узлы можно передавать `chat_history` (список сообщений роли/контента), он учитывается при формировании ответа.

Заметки
- Генератора SQL нет — предполагается внешний узел, который берёт `current_prompt`/`fixed_prompt` и выдаёт `sql_query`.
- Подключения к реальной БД нет — исполняющая логика реализована LLM для имитации/валидации запросов.
