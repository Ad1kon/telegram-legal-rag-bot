import asyncio
import logging

from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart
from aiogram.types import Message

from app.config import get_settings
from app.llm import answer_with_groq
from app.rag import build_prompt, search_context


settings = get_settings()
dp = Dispatcher()


@dp.message(CommandStart())
async def start(message: Message) -> None:
    await message.answer(
        "Здравствуйте. Задайте вопрос по загруженным конвенциям и юридическим документам."
    )


@dp.message(F.text)
async def handle_question(message: Message) -> None:
    if not settings.groq_api_key:
        await message.answer("Не задан GROQ_API_KEY в .env.")
        return

    await message.answer("Ищу по базе документов...")

    contexts = search_context(
        question=message.text or "",
        chroma_path=settings.chroma_path,
        collection_name=settings.collection_name,
        top_k=settings.top_k,
	anonymized_telemetry=settings.anonymized_telemetry,
    )

    if not contexts:
        await message.answer("В базе пока нет проиндексированных документов.")
        return

    prompt = build_prompt(message.text or "", contexts)
    answer = answer_with_groq(settings.groq_api_key, settings.groq_model, prompt)
    await message.answer(answer[:4000])


async def main() -> None:
    if not settings.telegram_bot_token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")

    logging.basicConfig(level=logging.INFO)
    bot = Bot(token=settings.telegram_bot_token)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
