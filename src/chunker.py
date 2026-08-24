import json
from collections import defaultdict
from pathlib import Path

DEFAULT_MAX_CHARS = 1400
DEFAULT_OVERLAP_CHARS = 250


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


def normalize_text(text):
    if text is None:
        return ""

    return " ".join(str(text).strip().split())


def make_page_key(record):
    return (
        record.get("root_folder", ""),
        record.get("folder", ""),
        record.get("image", ""),
        record.get("page_number", 0),
    )


def group_records_by_page(records):
    grouped = defaultdict(list)

    for record in records:
        grouped[make_page_key(record)].append(record)

    pages = []

    for page_key, page_records in grouped.items():
        page_records.sort(
            key=lambda item: (
                item.get("block_number", 0),
                item.get("top", 0) or 0,
            )
        )

        pages.append((page_key, page_records))

    pages.sort(
        key=lambda item: (
            item[0][0],
            item[0][1],
            item[0][3],
            item[0][2],
        )
    )

    return pages


def is_heading(block):
    return block.get("type") in {"heading", "subheading"}


def build_text_unit(block, current_heading):
    text = normalize_text(block.get("text", ""))

    if not text:
        return None

    return {
        "text": text,
        "type": block.get("type", "paragraph"),
        "block_number": block.get("block_number"),
        "heading": current_heading,
    }


def make_chunk(page_records, units, chunk_number, heading):
    first = page_records[0]

    texts = [
        unit["text"]
        for unit in units
        if unit.get("text")
    ]

    chunk_text = "\n".join(texts).strip()

    block_numbers = [
        unit.get("block_number")
        for unit in units
        if unit.get("block_number") is not None
    ]

    root_folder = first.get("root_folder", "")
    folder = first.get("folder", "")
    image = first.get("image", "")
    page_number = first.get("page_number")

    folder_part = (
        folder.replace("/", "_").replace("\\", "_")
        if folder
        else root_folder
    )

    image_stem = Path(image).stem

    chunk_id = (
        f"{folder_part}_{image_stem}_p{page_number}_c{chunk_number}"
    )

    return {
        "chunk_id": chunk_id,
        "root_folder": root_folder,
        "folder": folder,
        "folder_name": first.get("folder_name", ""),
        "image": image,
        "relative_image_path": first.get("relative_image_path", ""),
        "image_path": first.get("image_path", ""),
        "page_number": page_number,
        "chunk_number": chunk_number,
        "heading": heading,
        "block_start": min(block_numbers) if block_numbers else None,
        "block_end": max(block_numbers) if block_numbers else None,
        "text": chunk_text,
        "capture_time": first.get("capture_time"),
        "capture_time_source": first.get("capture_time_source"),
    }


def take_overlap_units(units, overlap_chars):
    if overlap_chars <= 0:
        return []

    overlap = []
    total_chars = 0

    for unit in reversed(units):
        overlap.insert(0, unit)
        total_chars += len(unit.get("text", ""))

        if total_chars >= overlap_chars:
            break

    return overlap


def chunk_page(
    page_records,
    max_chars=DEFAULT_MAX_CHARS,
    overlap_chars=DEFAULT_OVERLAP_CHARS,
):
    if not page_records:
        return []

    chunks = []
    current_heading = None
    current_units = []
    current_length = 0
    chunk_number = 1

    def flush_chunk():
        nonlocal current_units
        nonlocal current_length
        nonlocal chunk_number

        if not current_units:
            return

        chunk_heading = None

        for unit in current_units:
            if unit.get("type") == "heading":
                chunk_heading = unit.get("text")
            elif chunk_heading is None and unit.get("heading"):
                chunk_heading = unit.get("heading")

        chunk = make_chunk(
            page_records=page_records,
            units=current_units,
            chunk_number=chunk_number,
            heading=chunk_heading,
        )

        if chunk["text"]:
            chunks.append(chunk)
            chunk_number += 1

        overlap_units = take_overlap_units(
            current_units,
            overlap_chars,
        )

        if (
            overlap_units
            and all(
                unit.get("type") in {"heading", "subheading"}
                for unit in overlap_units
            )
        ):
            overlap_units = []

        current_units = overlap_units

        current_length = sum(
            len(unit.get("text", "")) + 1
            for unit in current_units
        )

    for block in page_records:
        text = normalize_text(block.get("text", ""))

        if not text:
            continue

        block_type = block.get("type", "paragraph")

        if block_type == "page_number":
            continue

        if is_heading(block):
            if block_type == "heading" and current_units:
                flush_chunk()

            if block_type == "heading":
                current_heading = text

            unit = build_text_unit(block, current_heading)

            if unit is None:
                continue

            projected_length = current_length + len(text) + 1

            if current_units and projected_length > max_chars:
                flush_chunk()

            current_units.append(unit)
            current_length += len(text) + 1
            continue

        unit = build_text_unit(block, current_heading)

        if unit is None:
            continue

        projected_length = current_length + len(text) + 1

        if current_units and projected_length > max_chars:
            flush_chunk()

        current_units.append(unit)
        current_length += len(text) + 1

    flush_chunk()

    return chunks


def build_chunks(
    records,
    max_chars=DEFAULT_MAX_CHARS,
    overlap_chars=DEFAULT_OVERLAP_CHARS,
):
    all_chunks = []

    for _, page_records in group_records_by_page(records):
        all_chunks.extend(
            chunk_page(
                page_records=page_records,
                max_chars=max_chars,
                overlap_chars=overlap_chars,
            )
        )

    return all_chunks


def build_chunks_from_file(
    input_path,
    output_path,
    max_chars=DEFAULT_MAX_CHARS,
    overlap_chars=DEFAULT_OVERLAP_CHARS,
):
    records = load_jsonl(input_path)

    if not records:
        raise ValueError(
            f"No OCR records were found in: {input_path}"
        )

    chunks = build_chunks(
        records=records,
        max_chars=max_chars,
        overlap_chars=overlap_chars,
    )

    save_jsonl(chunks, output_path)

    return {
        "ocr_records": len(records),
        "chunks": len(chunks),
        "output_path": str(Path(output_path).resolve()),
    }
