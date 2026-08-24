import json
from pathlib import Path


def _safe_relative_path(path, root_directory):
    path = Path(path)
    root_directory = Path(root_directory)

    try:
        return path.relative_to(root_directory)
    except ValueError:
        return path


def create_knowledge_record(
    block,
    source_image,
    page_number,
    folder_path,
    root_directory,
    block_number,
    capture_time=None,
    time_source=None,
    page_width=None,
    page_height=None
):
    """
    Convert one classified OCR block into a JSON-safe
    knowledge-base record.
    """
    source_image = Path(source_image)
    folder_path = Path(folder_path)
    root_directory = Path(root_directory)

    relative_folder = _safe_relative_path(
        folder_path,
        root_directory
    )

    relative_image = _safe_relative_path(
        source_image,
        root_directory
    )

    folder_text = (
        ""
        if str(relative_folder) == "."
        else relative_folder.as_posix()
    )

    record = {
        "root_folder": root_directory.name,
        "folder": folder_text,
        "folder_name": folder_path.name,

        "image": source_image.name,
        "relative_image_path": relative_image.as_posix(),
        "image_path": str(source_image),

        "page_number": page_number,
        "block_number": block_number,

        "type": block.get(
            "type",
            "paragraph"
        ),

        "text": str(
            block.get(
                "text",
                ""
            )
        ).strip(),

        "confidence": block.get(
            "confidence"
        ),

        "left": block.get(
            "left"
        ),
        "top": block.get(
            "top"
        ),
        "right": block.get(
            "right"
        ),
        "bottom": block.get(
            "bottom"
        ),

        "page_width": page_width,
        "page_height": page_height,

        "capture_time": (
            capture_time.isoformat()
            if hasattr(capture_time, "isoformat")
            else capture_time
        ),

        "capture_time_source": time_source
    }

    return record


def export_page_blocks(
    classified_blocks,
    source_image,
    page_number,
    folder_path,
    root_directory,
    capture_time=None,
    time_source=None,
    page_width=None,
    page_height=None
):
    """
    Convert useful OCR blocks from one page into records.

    We skip only page numbers.

    Text inside detected tables and figures is kept in the
    knowledge base because it may still be useful for later
    retrieval and question answering.
    """
    records = []

    for block_number, block in enumerate(
        classified_blocks,
        start=1
    ):
        text = str(
            block.get(
                "text",
                ""
            )
        ).strip()

        if not text:
            continue

        block_type = block.get(
            "type",
            "paragraph"
        )

        if block_type == "page_number":
            continue

        record = create_knowledge_record(
            block=block,
            source_image=source_image,
            page_number=page_number,
            folder_path=folder_path,
            root_directory=root_directory,
            block_number=block_number,
            capture_time=capture_time,
            time_source=time_source,
            page_width=page_width,
            page_height=page_height
        )

        records.append(
            record
        )

    return records


def save_knowledge_records(
    records,
    output_path
):
    """
    Save one JSON object per line.

    Existing content is replaced so each complete OCR run
    produces a fresh knowledge base.
    """
    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with output_path.open(
        "w",
        encoding="utf-8"
    ) as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                )
            )
            file.write("\n")

    return output_path