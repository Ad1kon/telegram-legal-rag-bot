from pathlib import Path
import sys
from collections import Counter

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.config import get_settings
from app.vectorstore import get_collection


def main() -> None:
    settings = get_settings()
    raw_path = Path(settings.raw_docs_path)
    pdf_files = sorted(raw_path.glob("*.pdf"))

    collection = get_collection(
        settings.chroma_path,
        settings.collection_name,
        settings.anonymized_telemetry,
    )

    total_chunks = collection.count()
    print(f"PDF files in {raw_path}: {len(pdf_files)}")
    for file in pdf_files:
        print(f"  - {file.name}")

    print(f"\nChunks in Chroma collection '{settings.collection_name}': {total_chunks}")

    if total_chunks == 0:
        print("\nIndex is empty. Run: python scripts/ingest.py")
        return

    rows = collection.get(include=["metadatas"], limit=total_chunks)
    metadatas = rows.get("metadatas", [])
    by_source = Counter(item.get("source", "unknown") for item in metadatas)

    print("\nChunks by source:")
    for source, count in sorted(by_source.items()):
        print(f"  - {source}: {count}")

    indexed_sources = set(by_source)
    raw_sources = {file.name for file in pdf_files}
    missing = sorted(raw_sources - indexed_sources)

    if missing:
        print("\nWARNING: These PDF files are present in data/raw but not found in Chroma metadata:")
        for source in missing:
            print(f"  - {source}")
    else:
        print("\nOK: every PDF file from data/raw has at least one chunk in Chroma.")


if __name__ == "__main__":
    main()
