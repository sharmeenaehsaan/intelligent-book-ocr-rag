from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
import traceback
from PIL import Image, UnidentifiedImageError

from src.preprocess import ImagePreprocessor
from src.ocr import OCREngine
from src.layout import classify_text_blocks

from src.figure_detector import (
    detect_figure_regions,
    save_figure_crops,
    mark_blocks_inside_figures
)

from src.table_detector import (
    detect_table_regions,
    mark_blocks_inside_tables,
    save_table_crops
)

from src.multi_page_word_generator import (
    build_multi_page_document
)

from src.knowledge_exporter import (
    export_page_blocks,
    save_knowledge_records
)


# =========================================================
# ROOT FOLDER SETTING
# =========================================================

# Change only this path.
#
# Example:
# World/
#   world_page_1.jpg
#   India/
#       india_page_1.jpg
#       Kashmir/
#           kashmir_page_1.jpg
#
# Results:
# World/World.docx
# World/India/India.docx
# World/India/Kashmir/Kashmir.docx

ROOT_DIRECTORY = Path(
    r"C:\Users\sharmeen\OneDrive\Desktop"
    r"\A"
)


PROJECT_DIRECTORY = Path(__file__).resolve().parent

KNOWLEDGE_BASE_DIRECTORY = (
    PROJECT_DIRECTORY / "knowledge_base"
)

KNOWLEDGE_OUTPUT_PATH = (
    KNOWLEDGE_BASE_DIRECTORY
    / "ocr_blocks.jsonl"
)


# =========================================================
# DETECTION SETTINGS
# =========================================================

ENABLE_FIGURE_DETECTION = True
ENABLE_TABLE_DETECTION = True

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
}

# Old output folders, virtual environments and technical folders
# are ignored during recursive processing.
SKIPPED_FOLDER_NAMES = {
    "__pycache__",
    ".git",
    ".venv",
    ".venv-1",
    "output",
    "figures",
    "tables",
    "visualizations"
}


# =========================================================
# IMAGE ORDER
# =========================================================

EXIF_DATE_TIME_ORIGINAL = 36867
EXIF_DATE_TIME_DIGITIZED = 36868
EXIF_DATE_TIME = 306


def parse_exif_datetime(value):
    if value is None:
        return None

    if isinstance(value, bytes):
        value = value.decode(
            "utf-8",
            errors="ignore"
        )

    value = str(value).strip().strip("\x00")

    formats = [
        "%Y:%m:%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S"
    ]

    for date_format in formats:
        try:
            return datetime.strptime(
                value[:19],
                date_format
            )
        except ValueError:
            continue

    return None


def get_image_capture_time(image_path):
    """
    Prefer the photograph's EXIF capture time.

    For PNG files, screenshots, edited images or images whose
    EXIF data was removed, use file modified time as a fallback.
    """
    try:
        with Image.open(image_path) as image:
            exif = image.getexif()

            if exif:
                date_tags = [
                    (
                        EXIF_DATE_TIME_ORIGINAL,
                        "EXIF DateTimeOriginal"
                    ),
                    (
                        EXIF_DATE_TIME_DIGITIZED,
                        "EXIF DateTimeDigitized"
                    ),
                    (
                        EXIF_DATE_TIME,
                        "EXIF DateTime"
                    )
                ]

                for tag_id, source_name in date_tags:
                    capture_time = parse_exif_datetime(
                        exif.get(tag_id)
                    )

                    if capture_time is not None:
                        return (
                            capture_time,
                            source_name
                        )

    except (
        UnidentifiedImageError,
        OSError,
        ValueError
    ):
        pass

    modified_time = datetime.fromtimestamp(
        image_path.stat().st_mtime
    )

    return (
        modified_time,
        "file modified time"
    )


def get_images_in_folder(directory):
    """
    Return only images located directly inside this folder.

    Images in child folders are not included here. Each child
    folder is processed separately by the recursive walker.
    """
    image_paths = [
        path
        for path in directory.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            in SUPPORTED_EXTENSIONS
        )
    ]

    image_records = []

    for image_path in image_paths:
        capture_time, time_source = (
            get_image_capture_time(
                image_path
            )
        )

        image_records.append({
            "path": image_path,
            "capture_time": capture_time,
            "time_source": time_source
        })

    # Oldest photographed page first.
    image_records.sort(
        key=lambda record: (
            record["capture_time"],
            record["path"].name.casefold()
        )
    )

    return image_records


# =========================================================
# FOLDER WALKING
# =========================================================

def should_skip_folder(directory):
    if directory.is_symlink():
        return True

    if directory.name.startswith("."):
        return True

    return (
        directory.name.casefold()
        in {
            name.casefold()
            for name in SKIPPED_FOLDER_NAMES
        }
    )


def get_child_folders(directory):
    child_folders = [
        path
        for path in directory.iterdir()
        if (
            path.is_dir()
            and not should_skip_folder(path)
        )
    ]

    child_folders.sort(
        key=lambda path: path.name.casefold()
    )

    return child_folders


def get_word_output_path(directory):
    """
    World folder   -> World/World.docx
    India folder   -> India/India.docx
    Kashmir folder -> Kashmir/Kashmir.docx
    """
    folder_name = directory.name.strip()

    if not folder_name:
        folder_name = "complete_book"

    return (
        directory
        / f"{folder_name}.docx"
    )


# =========================================================
# PROCESS ONE PAGE
# =========================================================

def process_single_page(
    image_path,
    page_number,
    processor,
    ocr_engine,
    table_work_directory,
    figure_work_directory
):
    print(
        f"\n{'=' * 60}\n"
        f"Processing page {page_number}: "
        f"{image_path.name}\n"
        f"{'=' * 60}"
    )

    # -----------------------------------------------------
    # 1. Preprocessing
    # -----------------------------------------------------

    processed_images = processor.preprocess(
        str(image_path)
    )

    original_image = processed_images[
        "original"
    ]

    # -----------------------------------------------------
    # 2. OCR
    # -----------------------------------------------------

    ocr_results = ocr_engine.extract(
        original_image
    )

    print("\nRAW OCR RESULTS:\n")

    for index, item in enumerate(
        ocr_results,
        start=1
    ):
        print(
            f"{index:03d} | "
            f"{item['confidence']:.2%} | "
            f"top={item['top']:.0f} | "
            f"{item['text']}"
        )

    print(
        f"OCR text regions: "
        f"{len(ocr_results)}"
    )

    if not ocr_results:
        print(
            "Warning: No text was detected "
            "on this page."
        )

    # -----------------------------------------------------
    # 3. Layout classification
    # -----------------------------------------------------

    classified_blocks = classify_text_blocks(
        ocr_results,
        page_height=original_image.shape[0],
        page_width=original_image.shape[1]
    )

    print("\nBLOCK TYPES:\n")

    for block in classified_blocks:
        print(
            f"{block['type']:15} | "
            f"top={block['top']:4} | "
            f"{block['text'][:80]}"
        )

    # -----------------------------------------------------
    # 4. Table detection
    # Tables are detected before figures.
    # -----------------------------------------------------

    if ENABLE_TABLE_DETECTION:
        table_boxes = detect_table_regions(
            original_image
        )
    else:
        table_boxes = []

    print(
        f"Tables detected: "
        f"{len(table_boxes)}"
    )

    page_table_directory = (
        table_work_directory
        / f"page_{page_number:03d}"
    )

    saved_tables = save_table_crops(
        original_image,
        table_boxes,
        page_table_directory,
        padding=8
    )

    if table_boxes:
        classified_blocks = (
            mark_blocks_inside_tables(
                classified_blocks,
                table_boxes
            )
        )

    # -----------------------------------------------------
    # 5. Figure detection
    # -----------------------------------------------------

    if ENABLE_FIGURE_DETECTION:
        figure_boxes = detect_figure_regions(
            original_image,
            classified_blocks,
            debug=False
        )
    else:
        figure_boxes = []

    print(
        f"Figures detected: "
        f"{len(figure_boxes)}"
    )

    page_figure_directory = (
        figure_work_directory
        / f"page_{page_number:03d}"
    )

    saved_figures = save_figure_crops(
        original_image,
        figure_boxes,
        page_figure_directory,
        padding=8
    )

    if figure_boxes:
        classified_blocks = (
            mark_blocks_inside_figures(
                classified_blocks,
                figure_boxes
            )
        )

    # -----------------------------------------------------
    # 6. Block summary
    # -----------------------------------------------------

    block_counts = {}

    for block in classified_blocks:
        block_type = block["type"]

        block_counts[block_type] = (
            block_counts.get(block_type, 0)
            + 1
        )

    print("\nBlock summary:")

    for block_type, count in sorted(
        block_counts.items()
    ):
        print(
            f"  {block_type:16}: {count}"
        )

    # -----------------------------------------------------
    # 7. Return page data
    # -----------------------------------------------------

    return {
        "page_number": page_number,
        "source_image": str(image_path),

        "page_width": original_image.shape[1],
        "page_height": original_image.shape[0],

        "classified_blocks": classified_blocks,

        "saved_tables": saved_tables,
        "table_boxes": table_boxes,

        "saved_figures": saved_figures,
        "figure_boxes": figure_boxes
    }


# =========================================================
# PROCESS IMAGES DIRECTLY INSIDE ONE FOLDER
# =========================================================

def process_folder_images(
    directory,
    processor,
    ocr_engine,
    root_directory,
    knowledge_records
):
    image_records = get_images_in_folder(
        directory
    )

    if not image_records:
        print(
            f"\nNo direct images in: "
            f"{directory}"
        )

        return None

    word_output_path = get_word_output_path(
        directory
    )

    print(
        "\n"
        + "#" * 70
    )

    print(
        f"FOLDER: {directory}"
    )

    print(
        f"Images found: "
        f"{len(image_records)}"
    )

    print(
        f"Word output: "
        f"{word_output_path}"
    )

    print("\nImage capture order:")

    for index, record in enumerate(
        image_records,
        start=1
    ):
        print(
            f"{index}. "
            f"{record['path'].name} | "
            f"{record['capture_time']:%Y-%m-%d %H:%M:%S} | "
            f"{record['time_source']}"
        )

    print(
        "#" * 70
    )

    processed_pages = []

    # Table and figure crops are required temporarily so they can
    # be inserted into Word. They are stored in the Windows temp
    # directory and automatically deleted after the Word file is saved.
    with TemporaryDirectory(
        prefix="book_to_word_"
    ) as temporary_directory:

        temporary_root = Path(
            temporary_directory
        )

        table_work_directory = (
            temporary_root / "tables"
        )

        figure_work_directory = (
            temporary_root / "figures"
        )

        table_work_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        figure_work_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        for page_number, record in enumerate(
            image_records,
            start=1
        ):
            page_data = process_single_page(
                image_path=record["path"],
                page_number=page_number,
                processor=processor,
                ocr_engine=ocr_engine,
                table_work_directory=(
                    table_work_directory
                ),
                figure_work_directory=(
                    figure_work_directory
                )
            )

            processed_pages.append(
                page_data
            )

            page_knowledge_records = export_page_blocks(
                classified_blocks=page_data[
                    "classified_blocks"
                ],
                source_image=record["path"],
                page_number=page_number,
                folder_path=directory,
                root_directory=root_directory,
                capture_time=record[
                    "capture_time"
                ],
                time_source=record[
                    "time_source"
                ],
                page_width=page_data[
                    "page_width"
                ],
                page_height=page_data[
                    "page_height"
                ]
            )

            knowledge_records.extend(
                page_knowledge_records
            )

        build_multi_page_document(
            processed_pages=processed_pages,
            output_path=word_output_path
        )

    print(
        f"\nSaved Word document:\n"
        f"{word_output_path}"
    )

    return word_output_path


# =========================================================
# RECURSIVE PROCESSING
# =========================================================

def process_folder_tree(
    directory,
    processor,
    ocr_engine,
    root_directory,
    knowledge_records,
    created_documents,
    failed_folders,
    visited_folders
):
    """
    Processing order:

    1. Process images directly inside the current folder.
    2. Save that folder's Word document inside the same folder.
    3. Enter each child folder and repeat the same process.
    """
    visited_folders.append(
        directory
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        f"Checking folder:\n{directory}"
    )

    print(
        "=" * 70
    )

    try:
        created_document = process_folder_images(
            directory=directory,
            processor=processor,
            ocr_engine=ocr_engine,
            root_directory=root_directory,
            knowledge_records=knowledge_records
        )

        if created_document is not None:
            created_documents.append(
                created_document
            )

    except Exception as error:
        failed_folders.append(
            (
                directory,
                str(error)
            )
        )

        print(
            f"\nERROR while processing folder:\n"
            f"{directory}"
        )

        print(error)
        traceback.print_exc()

    # Even if the current folder has a bad image, continue into
    # its child folders so the remaining tree can still be processed.
    for child_directory in get_child_folders(
        directory
    ):
        process_folder_tree(
            directory=child_directory,
            processor=processor,
            ocr_engine=ocr_engine,
            root_directory=root_directory,
            knowledge_records=knowledge_records,
            created_documents=created_documents,
            failed_folders=failed_folders,
            visited_folders=visited_folders
        )


# =========================================================
# MAIN PROGRAM
# =========================================================

def main():
    root_directory = ROOT_DIRECTORY.expanduser().resolve()

    if not root_directory.exists():
        raise FileNotFoundError(
            f"Root directory was not found:\n"
            f"{root_directory}"
        )

    if not root_directory.is_dir():
        raise NotADirectoryError(
            f"The selected root path is not a folder:\n"
            f"{root_directory}"
        )

    print(
        "\nRecursive book-page processing"
    )

    print(
        f"Root folder:\n"
        f"{root_directory}"
    )

    print(
        "\nWord documents remain inside the image folders. "
        "Temporary table and figure crops are automatically deleted. "
        "OCR knowledge is saved separately in the project's knowledge_base folder."
    )

    # Load the expensive OCR and preprocessing objects only once.
    processor = ImagePreprocessor()
    ocr_engine = OCREngine()

    knowledge_records = []
    created_documents = []
    failed_folders = []
    visited_folders = []

    process_folder_tree(
        directory=root_directory,
        processor=processor,
        ocr_engine=ocr_engine,
        root_directory=root_directory,
        knowledge_records=knowledge_records,
        created_documents=created_documents,
        failed_folders=failed_folders,
        visited_folders=visited_folders
    )


    knowledge_path = save_knowledge_records(
        records=knowledge_records,
        output_path=KNOWLEDGE_OUTPUT_PATH
    )

    print(
        f"\nKnowledge records saved: "
        f"{len(knowledge_records)}"
    )

    print(
        f"Knowledge file:\n"
        f"{knowledge_path}"
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "RECURSIVE PROCESSING COMPLETED"
    )

    print(
        f"Folders checked: "
        f"{len(visited_folders)}"
    )

    print(
        f"Word documents created: "
        f"{len(created_documents)}"
    )

    for document_path in created_documents:
        print(
            f"  {document_path}"
        )

    if failed_folders:
        print(
            f"\nFolders with errors: "
            f"{len(failed_folders)}"
        )

        for folder_path, error_message in failed_folders:
            print(
                f"  {folder_path}\n"
                f"    {error_message}"
            )
    else:
        print(
            "\nNo folder-processing errors occurred."
        )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()