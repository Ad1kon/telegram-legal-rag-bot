# Telegram Legal RAG Bot

ИИ-бот для анализа конвенций МОТ, ТК РК и других юридических PDF через Telegram.

## Архитектура

```text
PDF files
  -> scripts/ingest.py
  -> text extraction + chunking
  -> embeddings
  -> ChromaDB
  -> Telegram bot
  -> Groq LLM answer with source citations
```

## Что умеет MVP

- Загружает PDF из папки `data/raw`.
- Извлекает текст и режет его на фрагменты.
- Сохраняет векторную базу в `data/chroma`.
- При вопросе пользователя ищет релевантные фрагменты.
- Отвечает через Groq и добавляет источники.

## Структура

```text
telegram_legal_bot/
  app/
    bot.py
    config.py
    llm.py
    rag.py
    vectorstore.py
  scripts/
    ingest.py
  data/
    raw/
    chroma/
  .env.example
  requirements.txt
```

## Быстрый старт

1. Создайте Telegram-бота через `@BotFather` и получите `TELEGRAM_BOT_TOKEN`.
2. Получите `GROQ_API_KEY`.
3. Скопируйте `.env.example` в `.env` и заполните значения.
4. Положите PDF-файлы в `data/raw`.
5. Установите зависимости:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

6. Индексируйте документы:

```bash
python scripts/ingest.py
```

7. Запустите бота:

```bash
python -m app.bot
```

## Рекомендованный режим ответа

Бот должен отвечать осторожно:

- не выдавать ответ как юридическую консультацию;
- ссылаться на найденные документы;
- явно говорить, если в базе нет достаточного основания;
- не придумывать номера статей и формулировки.

## Следующий этап

После MVP стоит добавить:

- метаданные документов: страна, дата ратификации, тип документа, язык;
- админ-команду `/reindex`;
- роли пользователей;
- логирование вопросов;
- тестовый набор юридических вопросов для проверки качества.
