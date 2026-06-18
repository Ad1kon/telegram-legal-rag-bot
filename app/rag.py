from app.vectorstore import get_collection, get_embeddings



def search_context(
    question: str,
    chroma_path: str,
    collection_name: str,
    top_k: int,
    anonymized_telemetry: bool = False,
) -> list[dict]:
    embeddings = get_embeddings()
    collection = get_collection(chroma_path, collection_name, anonymized_telemetry)
    query_embedding = embeddings.embed([question])[0]

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    contexts: list[dict] = []
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    for text, metadata, distance in zip(documents, metadatas, distances):
        contexts.append(
            {
                "text": text,
                "source": metadata.get("source", "unknown"),
                "page": metadata.get("page", "unknown"),
                "distance": distance,
            }
        )

    return contexts


def build_prompt(question: str, contexts: list[dict]) -> str:
    context_text = "\n\n".join(
        f"[Источник: {item['source']}, стр. {item['page']}]\n{item['text']}"
        for item in contexts
    )

    return f"""
Ты юридический аналитический ассистент по документам из базы пользователя.
Отвечай только на основании приведенного контекста.
Если данных недостаточно, прямо скажи: "В загруженных документах недостаточно оснований для ответа".
Не придумывай статьи, даты, номера конвенций или формулировки.
В конце ответа добавь список использованных источников.

Контекст:
{context_text}

Вопрос:
{question}
""".strip()
