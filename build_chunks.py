from pathlib import Path

from src.chunker import build_chunks_from_file


PROJECT_DIRECTORY = Path(__file__).resolve().parent

KNOWLEDGE_BASE_DIRECTORY = (
    PROJECT_DIRECTORY / "knowledge_base"
)

OCR_BLOCKS_PATH = (
    KNOWLEDGE_BASE_DIRECTORY / "ocr_blocks.jsonl"
)

CHUNKS_PATH = (
    KNOWLEDGE_BASE_DIRECTORY / "chunks.jsonl"
)


def main():
    print("\n" + "=" * 70)
    print("BUILDING RAG CHUNKS")
    print("=" * 70)

    print(f"\nInput:\n{OCR_BLOCKS_PATH}")

    if not OCR_BLOCKS_PATH.exists():
        raise FileNotFoundError(
            "ocr_blocks.jsonl was not found.\n"
            "Run main.py first so OCR knowledge records are created."
        )

    result = build_chunks_from_file(
        input_path=OCR_BLOCKS_PATH,
        output_path=CHUNKS_PATH,
        max_chars=1400,
        overlap_chars=250,
    )

    print(f"\nOCR records read: {result['ocr_records']}")
    print(f"Chunks created: {result['chunks']}")
    print(f"\nChunk file:\n{result['output_path']}")
    print("\nStage 2 complete.")


if __name__ == "__main__":
    main()
