import json
import re
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can",
    "did", "do", "does", "for", "from", "had", "has", "have",
    "how", "i", "in", "into", "is", "it", "its", "me", "of",
    "on", "or", "our", "that", "the", "their", "them", "there",
    "this", "to", "was", "were", "what", "when", "where", "which",
    "who", "why", "with", "you", "your",
}


def load_jsonl(path):
    path = Path(path)
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}: {error}"
                ) from error

    return records


def save_jsonl(records, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False))
            file.write("\n")

    return path


def build_faiss_index(
    chunks_path,
    output_directory,
    model_name=DEFAULT_MODEL_NAME,
):
    chunks_path = Path(chunks_path)
    output_directory = Path(output_directory)

    chunks = load_jsonl(chunks_path)

    if not chunks:
        raise ValueError(
            f"No chunks were found in: {chunks_path}"
        )

    texts = [
        str(chunk.get("text", "")).strip()
        for chunk in chunks
    ]

    if any(not text for text in texts):
        raise ValueError(
            "At least one chunk has empty text. "
            "Rebuild chunks.jsonl before creating the index."
        )

    print(f"\nLoading embedding model:\n{model_name}")

    model = SentenceTransformer(model_name)

    print(
        f"\nCreating embeddings for {len(texts)} chunks..."
    )

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32",
    )

    dimension = int(
        embeddings.shape[1]
    )

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    index_path = (
        output_directory
        / "chunks.faiss"
    )

    metadata_path = (
        output_directory
        / "chunk_metadata.jsonl"
    )

    manifest_path = (
        output_directory
        / "manifest.json"
    )

    faiss.write_index(
        index,
        str(index_path),
    )

    save_jsonl(
        chunks,
        metadata_path,
    )

    manifest = {
        "model_name": model_name,
        "embedding_dimension": dimension,
        "number_of_chunks": len(chunks),
        "index_type": "IndexFlatIP",
        "similarity": "cosine",
        "normalized_embeddings": True,
        "retrieval": "hybrid_semantic_lexical",
        "source_chunks_file": str(
            chunks_path.resolve()
        ),
    }

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return {
        "model_name": model_name,
        "dimension": dimension,
        "chunks": len(chunks),
        "index_path": str(
            index_path.resolve()
        ),
        "metadata_path": str(
            metadata_path.resolve()
        ),
        "manifest_path": str(
            manifest_path.resolve()
        ),
    }


def load_search_components(
    index_directory,
):
    index_directory = Path(
        index_directory
    )

    index_path = (
        index_directory
        / "chunks.faiss"
    )

    metadata_path = (
        index_directory
        / "chunk_metadata.jsonl"
    )

    manifest_path = (
        index_directory
        / "manifest.json"
    )

    for required_path in (
        index_path,
        metadata_path,
        manifest_path,
    ):
        if not required_path.exists():
            raise FileNotFoundError(
                f"Missing vector-index file: {required_path}"
            )

    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    metadata = load_jsonl(
        metadata_path
    )

    index = faiss.read_index(
        str(index_path)
    )

    if index.ntotal != len(metadata):
        raise ValueError(
            "FAISS index and metadata count do not match. "
            "Please rebuild the vector index."
        )

    model_name = manifest[
        "model_name"
    ]

    print(
        f"Loading embedding model: "
        f"{model_name}"
    )

    model = SentenceTransformer(
        model_name
    )

    return (
        index,
        metadata,
        model,
        manifest,
    )


def _tokens(text):
    return re.findall(
        r"[a-z0-9]+",
        str(text).lower(),
    )


def _important_tokens(text):
    return [
        token
        for token in _tokens(text)
        if token not in STOP_WORDS
        and len(token) > 1
    ]


def _lexical_score(
    question,
    chunk_text,
):
    """
    Score exact/keyword overlap.

    This fixes cases where an exact title or phrase exists in OCR
    but the embedding similarity alone is unexpectedly low.
    """
    query_tokens = _important_tokens(
        question
    )

    if not query_tokens:
        return 0.0

    text_tokens = _tokens(
        chunk_text
    )

    text_token_set = set(
        text_tokens
    )

    unique_query_tokens = list(
        dict.fromkeys(
            query_tokens
        )
    )

    matched = sum(
        1
        for token in unique_query_tokens
        if token in text_token_set
    )

    coverage = (
        matched
        / len(unique_query_tokens)
    )

    normalized_question = " ".join(
        _tokens(question)
    )

    normalized_text = " ".join(
        text_tokens
    )

    phrase_bonus = 0.0

    # Strong bonus when the complete normalized question
    # appears in the chunk.
    if (
        normalized_question
        and normalized_question
        in normalized_text
    ):
        phrase_bonus = 0.25

    # Also reward a consecutive sequence of important query terms.
    important_phrase = " ".join(
        unique_query_tokens
    )

    if (
        important_phrase
        and important_phrase
        in normalized_text
    ):
        phrase_bonus = max(
            phrase_bonus,
            0.20,
        )

    score = (
        0.80 * coverage
        + phrase_bonus
    )

    return min(
        score,
        1.0,
    )


def search_chunks(
    question,
    index,
    metadata,
    model,
    top_k=5,
):
    """
    Hybrid retrieval:
    1. semantic similarity from FAISS
    2. lexical/exact-term matching over chunk text
    3. combined reranking

    Returned `similarity_score` is the final hybrid score.
    """
    question = str(
        question
    ).strip()

    if not question:
        return []

    if not metadata:
        return []

    top_k = max(
        1,
        min(
            int(top_k),
            len(metadata),
        ),
    )

    query_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32",
    )

    semantic_candidate_count = min(
        len(metadata),
        max(
            top_k * 5,
            20,
        ),
    )

    semantic_scores, semantic_indices = index.search(
        query_embedding,
        semantic_candidate_count,
    )

    semantic_by_index = {}

    for score, index_position in zip(
        semantic_scores[0],
        semantic_indices[0],
    ):
        if index_position < 0:
            continue

        semantic_by_index[
            int(index_position)
        ] = float(score)

    candidates = []

    for index_position, chunk in enumerate(
        metadata
    ):
        semantic_score = semantic_by_index.get(
            index_position,
            0.0,
        )

        lexical_score = _lexical_score(
            question=question,
            chunk_text=chunk.get(
                "text",
                ""
            ),
        )

        # Keep chunks that have either semantic or lexical evidence.
        if (
            semantic_score <= 0.0
            and lexical_score <= 0.0
        ):
            continue

        # Semantic retrieval remains primary, while lexical
        # matching rescues exact titles, names and phrases.
        hybrid_score = (
            0.75 * semantic_score
            + 0.25 * lexical_score
        )

        # A near-complete lexical match should not be discarded
        # simply because OCR wording lowered embedding similarity.
        if lexical_score >= 0.75:
            hybrid_score += 0.08

        hybrid_score = min(
            hybrid_score,
            1.0,
        )

        result = dict(
            chunk
        )

        result[
            "semantic_score"
        ] = float(
            semantic_score
        )

        result[
            "lexical_score"
        ] = float(
            lexical_score
        )

        result[
            "similarity_score"
        ] = float(
            hybrid_score
        )

        candidates.append(
            result
        )

    candidates.sort(
        key=lambda item: item[
            "similarity_score"
        ],
        reverse=True,
    )

    results = candidates[
        :top_k
    ]

    for rank, result in enumerate(
        results,
        start=1,
    ):
        result[
            "rank"
        ] = rank

    return results
