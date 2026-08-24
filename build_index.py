from pathlib import Path

from src.vector_index import build_faiss_index


PROJECT_DIRECTORY = Path(__file__).resolve().parent

KNOWLEDGE_BASE_DIRECTORY = (
    PROJECT_DIRECTORY / "knowledge_base"
)

CHUNKS_PATH = (
    KNOWLEDGE_BASE_DIRECTORY / "chunks.jsonl"
)

VECTOR_INDEX_DIRECTORY = (
    KNOWLEDGE_BASE_DIRECTORY / "vector_index"
)


def main():
    print("\n" + "=" * 70)
    print("BUILDING EMBEDDINGS + FAISS VECTOR INDEX")
    print("=" * 70)

    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            "chunks.jsonl was not found.\n"
            "Run: python build_chunks.py"
        )

    result = build_faiss_index(
        chunks_path=CHUNKS_PATH,
        output_directory=VECTOR_INDEX_DIRECTORY,
    )

    print("\n" + "=" * 70)
    print("VECTOR INDEX CREATED")
    print("=" * 70)

    print(f"Chunks indexed: {result['chunks']}")
    print(f"Embedding dimension: {result['dimension']}")
    print(f"Model: {result['model_name']}")
    print(f"\nFAISS index:\n{result['index_path']}")
    print(f"\nMetadata:\n{result['metadata_path']}")
    print(f"\nManifest:\n{result['manifest_path']}")


if __name__ == "__main__":
    main()
