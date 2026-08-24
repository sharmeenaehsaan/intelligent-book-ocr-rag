import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from src.vector_index import (
    load_search_components,
    search_chunks,
)


PROJECT_DIRECTORY = Path(__file__).resolve().parent
ENV_PATH = PROJECT_DIRECTORY / ".env"

VECTOR_INDEX_DIRECTORY = (
    PROJECT_DIRECTORY
    / "knowledge_base"
    / "vector_index"
)


def load_settings():
    load_dotenv(
        dotenv_path=ENV_PATH
    )

    api_key = os.getenv(
        "GROQ_API_KEY",
        ""
    ).strip()

    model_name = os.getenv(
        "GROQ_MODEL",
        "openai/gpt-oss-120b"
    ).strip()

    top_k = int(
        os.getenv(
            "TOP_K",
            "5"
        )
    )

    min_similarity = float(
        os.getenv(
            "MIN_SIMILARITY",
            "0.30"
        )
    )

    max_context_chunks = int(
        os.getenv(
            "MAX_CONTEXT_CHUNKS",
            "3"
        )
    )

    if (
        not api_key
        or api_key
        == "PASTE_YOUR_GROQ_API_KEY_HERE"
    ):
        raise RuntimeError(
            "\nGroq API key is missing.\n"
            "Open .env and add your real GROQ_API_KEY."
        )

    return {
        "api_key": api_key,
        "model_name": model_name,
        "top_k": top_k,
        "min_similarity": min_similarity,
        "max_context_chunks": max_context_chunks,
    }


def choose_context_chunks(
    search_results,
    min_similarity,
    max_context_chunks,
):
    """
    Select strong hybrid-retrieval results.

    If nothing reaches the normal cutoff, allow one weaker
    result only when lexical evidence is strong. This helps
    exact names/titles such as "A Tear and a Smile".
    """
    selected = []

    for result in search_results:
        score = float(
            result.get(
                "similarity_score",
                0.0
            )
        )

        if score < min_similarity:
            continue

        selected.append(
            result
        )

        if len(selected) >= max_context_chunks:
            break

    if selected:
        return selected

    # Safe fallback for an exact/strong keyword match.
    if search_results:
        best = search_results[0]

        lexical_score = float(
            best.get(
                "lexical_score",
                0.0
            )
        )

        semantic_score = float(
            best.get(
                "semantic_score",
                0.0
            )
        )

        if (
            lexical_score >= 0.60
            or semantic_score >= 0.24
        ):
            return [
                best
            ]

    return []


def build_context(chunks):
    sections = []

    for number, chunk in enumerate(
        chunks,
        start=1
    ):
        source_image = (
            chunk.get(
                "relative_image_path"
            )
            or chunk.get(
                "image"
            )
            or "unknown image"
        )

        page_number = chunk.get(
            "page_number"
        )

        heading = chunk.get(
            "heading"
        )

        text = chunk.get(
            "text",
            ""
        ).strip()

        header = (
            f"[SOURCE {number}]\n"
            f"Image: {source_image}\n"
            f"Page: {page_number}\n"
        )

        if heading:
            header += (
                f"Heading: {heading}\n"
            )

        sections.append(
            header
            + "Text:\n"
            + text
        )

    return "\n\n".join(
        sections
    )


def build_source_list(chunks):
    sources = []
    seen = set()

    for chunk in chunks:
        source_image = (
            chunk.get(
                "relative_image_path"
            )
            or chunk.get(
                "image"
            )
            or "unknown image"
        )

        page_number = chunk.get(
            "page_number"
        )

        key = (
            source_image,
            page_number
        )

        if key in seen:
            continue

        seen.add(
            key
        )

        sources.append(
            {
                "image": source_image,
                "page_number": page_number,
            }
        )

    return sources


def ask_groq(
    client,
    model_name,
    question,
    context,
):
    system_prompt = (
        "You are a question-answering assistant for OCR-extracted "
        "book pages.\n\n"
        "RULES:\n"
        "1. Answer ONLY from the supplied OCR context.\n"
        "2. Do not use outside knowledge to fill missing information.\n"
        "3. OCR text can contain spelling and punctuation errors. "
        "Interpret obvious OCR errors cautiously, but do not invent facts.\n"
        "4. If the supplied context does not contain enough information "
        "to answer the question, say: "
        "\"I could not find enough information in the scanned pages to answer that question.\"\n"
        "5. Give a clear and direct answer.\n"
        "6. Do not invent source names or page numbers.\n"
        "7. Do not add a Sources section yourself. "
        "The application displays sources separately."
    )

    user_prompt = (
        f"QUESTION:\n{question}\n\n"
        f"OCR CONTEXT:\n{context}"
    )

    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
    )

    answer = (
        completion
        .choices[0]
        .message
        .content
    )

    if not answer:
        return (
            "The model returned an empty answer."
        )

    return answer.strip()


def print_retrieval_debug(chunks):
    print(
        "\nRetrieved context:"
    )

    for number, chunk in enumerate(
        chunks,
        start=1
    ):
        image = (
            chunk.get(
                "relative_image_path"
            )
            or chunk.get(
                "image"
            )
        )

        hybrid = float(
            chunk.get(
                "similarity_score",
                0.0
            )
        )

        semantic = float(
            chunk.get(
                "semantic_score",
                0.0
            )
        )

        lexical = float(
            chunk.get(
                "lexical_score",
                0.0
            )
        )

        print(
            f"  {number}. "
            f"{image} | "
            f"page {chunk.get('page_number')} | "
            f"hybrid {hybrid:.4f} | "
            f"semantic {semantic:.4f} | "
            f"lexical {lexical:.4f}"
        )


def main():
    print(
        "\n"
        + "=" * 70
    )

    print(
        "BOOK OCR QUESTION ANSWERING"
    )

    print(
        "=" * 70
    )

    if not ENV_PATH.exists():
        raise FileNotFoundError(
            f".env file was not found:\n{ENV_PATH}"
        )

    if not VECTOR_INDEX_DIRECTORY.exists():
        raise FileNotFoundError(
            "\nVector index was not found.\n"
            "Run:\n"
            "python build_index.py\n"
        )

    settings = load_settings()

    print(
        "\nLoading local semantic-search system..."
    )

    (
        index,
        metadata,
        embedding_model,
        manifest,
    ) = load_search_components(
        VECTOR_INDEX_DIRECTORY
    )

    print(
        f"\nReady."
        f"\nIndexed chunks: {index.ntotal}"
        f"\nLLM: {settings['model_name']}"
    )

    print(
        "\nAsk questions about your scanned pages."
    )

    print(
        "Type 'exit' to stop."
    )

    client = Groq(
        api_key=settings[
            "api_key"
        ]
    )

    while True:
        question = input(
            "\nQuestion: "
        ).strip()

        if question.lower() in {
            "exit",
            "quit",
            "q",
        }:
            print(
                "\nGoodbye."
            )
            break

        if not question:
            continue

        search_results = search_chunks(
            question=question,
            index=index,
            metadata=metadata,
            model=embedding_model,
            top_k=settings[
                "top_k"
            ],
        )

        selected_chunks = choose_context_chunks(
            search_results=search_results,
            min_similarity=settings[
                "min_similarity"
            ],
            max_context_chunks=settings[
                "max_context_chunks"
            ],
        )

        if not selected_chunks:
            print(
                "\nAnswer:\n"
                "I could not find enough information "
                "in the scanned pages to answer that question."
            )
            continue

        print_retrieval_debug(
            selected_chunks
        )

        context = build_context(
            selected_chunks
        )

        try:
            answer = ask_groq(
                client=client,
                model_name=settings[
                    "model_name"
                ],
                question=question,
                context=context,
            )
        except Exception as error:
            print(
                "\nGroq API request failed:"
            )
            print(
                error
            )
            continue

        print(
            "\n"
            + "=" * 70
        )

        print(
            "ANSWER"
        )

        print(
            "=" * 70
        )

        print(
            answer
        )

        sources = build_source_list(
            selected_chunks
        )

        print(
            "\nSources:"
        )

        for number, source in enumerate(
            sources,
            start=1
        ):
            print(
                f"{number}. "
                f"{source['image']} "
                f"— Page "
                f"{source['page_number']}"
            )


if __name__ == "__main__":
    main()
