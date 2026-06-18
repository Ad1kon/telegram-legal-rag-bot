from groq import Groq


def answer_with_groq(api_key: str, model: str, prompt: str) -> str:
    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Ты аккуратный юридический аналитик. Отвечай кратко, точно и с опорой на источники.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
    )
    return completion.choices[0].message.content or ""
