from pathlib import Path
import argparse
import sys
import uuid

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.config import get_settings
from app.vectorstore import get_client, get_collection, get_embeddings


def add_batch(collection, embeddings, ids, documents, metadatas) -> int:
    if not documents:
        return 0

    vectors = embeddings.embed(documents)
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=vectors,
    )
    return len(documents)


def extract_pdf_pages(path: Path) -> list[dict]:
    reader = PdfReader(str(path))
    pages = []

    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        cleaned = " ".join(text.split())
        if cleaned:
            pages.append({"text": cleaned, "source": path.name, "page": index})

    return pages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the existing Chroma collection before indexing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    raw_path = Path(settings.raw_docs_path)
    raw_path.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(raw_path.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {raw_path}")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=180,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    if args.reset:
        client = get_client(settings.chroma_path, settings.anonymized_telemetry)
        try:
            client.delete_collection(settings.collection_name)
            print(f"Deleted existing collection: {settings.collection_name}")
        except ValueError:
            print(f"Collection does not exist yet: {settings.collection_name}")

    embeddings = get_embeddings()
    collection = get_collection(
        settings.chroma_path,
        settings.collection_name,
        settings.anonymized_telemetry,
    )

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []
    total_chunks = 0

    for pdf_path in pdf_files:
        file_chunks = 0
        print(f"Processing {pdf_path.name}...", flush=True)
        try:
            pages = extract_pdf_pages(pdf_path)
        except Exception as exc:
            print(f"ERROR: failed to extract {pdf_path.name}: {exc}")
            continue

        for page in pages:
            chunks = splitter.split_text(page["text"])
            for chunk_index, chunk in enumerate(chunks):
                ids.append(str(uuid.uuid4()))
                documents.append(chunk)
                metadatas.append(
                    {
                        "source": page["source"],
                        "page": page["page"],
                        "chunk": chunk_index,
                    }
                )

                if len(documents) >= settings.ingest_batch_size:
                    added = add_batch(collection, embeddings, ids, documents, metadatas)
                    total_chunks += added
                    file_chunks += added
                    ids.clear()
                    documents.clear()
                    metadatas.clear()

        added = add_batch(collection, embeddings, ids, documents, metadatas)
        total_chunks += added
        file_chunks += added
        ids.clear()
        documents.clear()
        metadatas.clear()

        print(
            f"{pdf_path.name}: pages_with_text={len(pages)}, chunks={file_chunks}",
            flush=True,
        )

    if total_chunks == 0:
        print("No text chunks were extracted from PDF files.")
        return

    print(f"Indexed {total_chunks} chunks from {len(pdf_files)} PDF files.")


if __name__ == "__main__":
    main()
