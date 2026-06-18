from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.config import get_settings
from app.rag import build_prompt, search_context


def main() -> None:
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        print('Usage: python scripts/debug_query.py "your question"')
        return

    settings = get_settings()
    contexts = search_context(
        question=question,
        chroma_path=settings.chroma_path,
        collection_name=settings.collection_name,
        top_k=settings.top_k,
        anonymized_telemetry=settings.anonymized_telemetry,
    )

    print(f"Question: {question}")
    print(f"Retrieved chunks: {len(contexts)}\n")

    for index, item in enumerate(contexts, start=1):
        preview = item["text"][:700].replace("\n", " ")
        print(f"--- Chunk {index} ---")
        print(f"Source: {item['source']}")
        print(f"Page: {item['page']}")
        print(f"Distance: {item['distance']}")
        print(f"Preview: {preview}")
        print()

    prompt = build_prompt(question, contexts)
    print("--- Prompt size ---")
    print(f"Characters sent to Groq, approximately: {len(prompt)}")


if __name__ == "__main__":
    main()
