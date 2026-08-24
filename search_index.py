from pathlib import Path

from src.vector_index import (
    load_search_components,
    search_chunks,
)


PROJECT_DIRECTORY = Path(__file__).resolve().parent

VECTOR_INDEX_DIRECTORY = (
    PROJECT_DIRECTORY
    / "knowledge_base"
    / "vector_index"
)


def print_result(result):
    print("\n" + "-" * 70)
    print(
        f"Rank: {result['rank']} | "
        f"Similarity: {result['similarity_score']:.4f}"
    )

    heading = result.get("heading")
    if heading:
        print(f"Heading: {heading}")

    print(
        f"Source image: "
        f"{result.get('relative_image_path') or result.get('image')}"
    )

    print(
        f"Page number: "
        f"{result.get('page_number')}"
    )

    print("\nText:")
    print(result.get("text", ""))


def main():
    print("\n" + "=" * 70)
    print("LOCAL SEMANTIC SEARCH TEST")
    print("=" * 70)

    index, metadata, model, manifest = load_search_components(
        VECTOR_INDEX_DIRECTORY
    )

    print(
        f"\nReady. Indexed chunks: {index.ntotal}"
    )

    print(
        "Type a question about the OCR pages."
    )

    print(
        "Type 'exit' to stop."
    )

    while True:
        question = input("\nQuestion: ").strip()

        if question.lower() in {
            "exit",
            "quit",
            "q",
        }:
            print("Goodbye.")
            break

        if not question:
            continue

        results = search_chunks(
            question=question,
            index=index,
            metadata=metadata,
            model=model,
            top_k=5,
        )

        if not results:
            print(
                "No relevant chunks found."
            )
            continue

        for result in results:
            print_result(result)


if __name__ == "__main__":
    main()
