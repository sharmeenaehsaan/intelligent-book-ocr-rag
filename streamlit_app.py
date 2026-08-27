from pathlib import Path
from tempfile import TemporaryDirectory
import shutil
import torch

import streamlit as st
from groq import Groq

from src.preprocess import ImagePreprocessor
from src.ocr import OCREngine
from src.layout import classify_text_blocks
from src.multi_page_word_generator import build_multi_page_document
from src.knowledge_exporter import export_page_blocks, save_knowledge_records
from src.chunker import build_chunks_from_file
from src.vector_index import build_faiss_index, load_search_components, search_chunks

# Reuse the project's existing full page pipeline only for Word conversion.
from main import process_single_page, get_image_capture_time

# Reuse the existing retrieval/Groq logic from ask.py.
from ask import (
    load_settings,
    choose_context_chunks,
    build_context,
    build_source_list,
    ask_groq,
)

PROJECT_DIRECTORY = Path(__file__).resolve().parent
RUNTIME_DIRECTORY = PROJECT_DIRECTORY / "runtime"
WORD_JOB_DIRECTORY = RUNTIME_DIRECTORY / "current_word_job"
LAST_RAG_UPLOAD_DIRECTORY = RUNTIME_DIRECTORY / "last_rag_upload"
KNOWLEDGE_BASE_DIRECTORY = PROJECT_DIRECTORY / "knowledge_base"
OCR_BLOCKS_PATH = KNOWLEDGE_BASE_DIRECTORY / "ocr_blocks.jsonl"
CHUNKS_PATH = KNOWLEDGE_BASE_DIRECTORY / "chunks.jsonl"
VECTOR_INDEX_DIRECTORY = KNOWLEDGE_BASE_DIRECTORY / "vector_index"

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

st.set_page_config(
    page_title="Intelligent Book OCR + RAG",
    page_icon="📚",
    layout="wide",
)

DEFAULT_STATE = {
    "screen": "home",
    "uploaded_files": None,
    "word_output_path": None,
    "messages": [],
    "rag_prepared": False,
    "rag_result": None,
}
for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


def go_to(screen):
    st.session_state.screen = screen
    st.rerun()


def reset_chat():
    st.session_state.messages = []


def clear_directory(directory):
    directory = Path(directory)
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)


def save_uploaded_files(uploaded_files, destination_directory):
    destination_directory = Path(destination_directory)
    clear_directory(destination_directory)

    saved_paths = []
    used_names = set()

    for uploaded_file in uploaded_files:
        original_name = Path(uploaded_file.name).name
        suffix = Path(original_name).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            continue

        stem = Path(original_name).stem
        candidate_name = original_name
        counter = 2
        while candidate_name.casefold() in used_names:
            candidate_name = f"{stem}_{counter}{suffix}"
            counter += 1

        used_names.add(candidate_name.casefold())
        output_path = destination_directory / candidate_name
        output_path.write_bytes(uploaded_file.getbuffer())
        saved_paths.append(output_path)

    return saved_paths


def sort_image_paths(image_paths):
    records = []
    for path in image_paths:
        capture_time, time_source = get_image_capture_time(path)
        records.append({
            "path": path,
            "capture_time": capture_time,
            "time_source": time_source,
        })

    records.sort(
        key=lambda item: (
            item["capture_time"],
            item["path"].name.casefold(),
        )
    )
    return records


def process_page_for_rag(image_path, page_number, processor, ocr_engine):
    """Only preprocessing + OCR + layout. No Word/table/figure pipeline."""
    processed_images = processor.preprocess(str(image_path))
    original_image = processed_images["original"]
    ocr_results = ocr_engine.extract(original_image)
    classified_blocks = classify_text_blocks(
        ocr_results,
        page_height=original_image.shape[0],
        page_width=original_image.shape[1],
    )
    return {
        "page_number": page_number,
        "source_image": str(image_path),
        "page_width": original_image.shape[1],
        "page_height": original_image.shape[0],
        "classified_blocks": classified_blocks,
    }


def convert_uploads_to_word(uploaded_files, progress_callback=None):
    """Word branch. Does not touch knowledge_base."""
    image_paths = save_uploaded_files(uploaded_files, WORD_JOB_DIRECTORY)
    if not image_paths:
        raise ValueError("No supported images were uploaded.")

    image_records = sort_image_paths(image_paths)
    processor = ImagePreprocessor()
    ocr_engine = OCREngine()
    processed_pages = []

    with TemporaryDirectory(prefix="book_streamlit_word_") as temporary_directory:
        temporary_root = Path(temporary_directory)
        table_work_directory = temporary_root / "tables"
        figure_work_directory = temporary_root / "figures"
        table_work_directory.mkdir(parents=True, exist_ok=True)
        figure_work_directory.mkdir(parents=True, exist_ok=True)

        total = len(image_records)
        for page_number, record in enumerate(image_records, start=1):
            if progress_callback:
                progress_callback(
                    page_number - 1,
                    total,
                    f"Processing page {page_number} of {total}: {record['path'].name}",
                )

            page_data = process_single_page(
                image_path=record["path"],
                page_number=page_number,
                processor=processor,
                ocr_engine=ocr_engine,
                table_work_directory=table_work_directory,
                figure_work_directory=figure_work_directory,
            )
            processed_pages.append(page_data)

        output_path = WORD_JOB_DIRECTORY / "converted_book.docx"
        build_multi_page_document(
            processed_pages=processed_pages,
            output_path=output_path,
        )

    if progress_callback:
        progress_callback(total, total, "Word document created.")
    return output_path


def prepare_uploads_for_questions(uploaded_files, progress_callback=None):
    """RAG branch. Replaces only the previous RAG upload and creates no Word file."""
    image_paths = save_uploaded_files(uploaded_files, LAST_RAG_UPLOAD_DIRECTORY)
    if not image_paths:
        raise ValueError("No supported images were uploaded.")

    image_records = sort_image_paths(image_paths)
    processor = ImagePreprocessor()
    ocr_engine = OCREngine()
    knowledge_records = []
    total = len(image_records)

    for page_number, record in enumerate(image_records, start=1):
        if progress_callback:
            progress_callback(
                page_number - 1,
                total + 2,
                f"OCR page {page_number} of {total}: {record['path'].name}",
            )

        page_data = process_page_for_rag(
            image_path=record["path"],
            page_number=page_number,
            processor=processor,
            ocr_engine=ocr_engine,
        )

        page_records = export_page_blocks(
            classified_blocks=page_data["classified_blocks"],
            source_image=record["path"],
            page_number=page_number,
            folder_path=LAST_RAG_UPLOAD_DIRECTORY,
            root_directory=LAST_RAG_UPLOAD_DIRECTORY,
            capture_time=record["capture_time"],
            time_source=record["time_source"],
            page_width=page_data["page_width"],
            page_height=page_data["page_height"],
        )
        knowledge_records.extend(page_records)

    if not knowledge_records:
        raise ValueError("OCR completed, but no usable text was found.")

    if progress_callback:
        progress_callback(total, total + 2, "Creating text chunks...")

    save_knowledge_records(knowledge_records, OCR_BLOCKS_PATH)
    chunk_result = build_chunks_from_file(
        input_path=OCR_BLOCKS_PATH,
        output_path=CHUNKS_PATH,
        max_chars=1400,
        overlap_chars=250,
    )

    if VECTOR_INDEX_DIRECTORY.exists():
        shutil.rmtree(VECTOR_INDEX_DIRECTORY)

    if progress_callback:
        progress_callback(total + 1, total + 2, "Creating embeddings + FAISS index...")

    index_result = build_faiss_index(
        chunks_path=CHUNKS_PATH,
        output_directory=VECTOR_INDEX_DIRECTORY,
    )

    if progress_callback:
        progress_callback(total + 2, total + 2, "Question-answering system ready.")

    return {
        "pages": len(image_records),
        "ocr_records": len(knowledge_records),
        "chunks": chunk_result["chunks"],
        "index_chunks": index_result["chunks"],
    }


def previous_rag_exists():
    required_files = [
        OCR_BLOCKS_PATH,
        CHUNKS_PATH,
        VECTOR_INDEX_DIRECTORY / "chunks.faiss",
        VECTOR_INDEX_DIRECTORY / "chunk_metadata.jsonl",
        VECTOR_INDEX_DIRECTORY / "manifest.json",
    ]
    return all(path.exists() for path in required_files)


@st.cache_resource(show_spinner=False)
def load_rag_resources(manifest_modified_time):
    settings = load_settings()
    index, metadata, embedding_model, manifest = load_search_components(
        VECTOR_INDEX_DIRECTORY
    )
    client = Groq(api_key=settings["api_key"])
    return settings, index, metadata, embedding_model, manifest, client


def answer_question(question):
    manifest_path = VECTOR_INDEX_DIRECTORY / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("No previous RAG index exists.")

    manifest_stamp = manifest_path.stat().st_mtime_ns
    settings, index, metadata, embedding_model, manifest, client = load_rag_resources(
        manifest_stamp
    )

    search_results = search_chunks(
        question=question,
        index=index,
        metadata=metadata,
        model=embedding_model,
        top_k=settings["top_k"],
    )

    selected_chunks = choose_context_chunks(
        search_results=search_results,
        min_similarity=settings["min_similarity"],
        max_context_chunks=settings["max_context_chunks"],
    )

    if not selected_chunks:
        return {
            "answer": "I could not find enough information in the scanned pages to answer that question.",
            "sources": [],
        }

    context = build_context(selected_chunks)
    answer = ask_groq(
        client=client,
        model_name=settings["model_name"],
        question=question,
        context=context,
    )
    return {
        "answer": answer,
        "sources": build_source_list(selected_chunks),
    }


def render_top_bar():
    left, right = st.columns([5, 1])
    with left:
        st.caption("OCR • Editable Word • Retrieval-Augmented Generation")
    with right:
        if st.session_state.screen != "home":
            if st.button("🏠 Home", use_container_width=True):
                go_to("home")


def render_home():
    st.title("📚 Intelligent Book OCR + RAG")
    st.subheader("Welcome to your intelligent book assistant")
    st.write(
        "Upload scanned or photographed book pages and choose exactly what you need: "
        "convert them into an editable Word document or ask questions from their content."
    )
    st.info("Only the processing required for the selected option is run.")

    st.markdown("### What would you like to do?")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📤 Upload New Images")
        st.write("Upload pages, then choose **Convert to Word** or **Ask Questions**.")
        if st.button("Upload New Images", type="primary", use_container_width=True):
            go_to("upload")

    with col2:
        st.markdown("#### 💬 Ask From Last Processed Upload")
        if previous_rag_exists():
            st.write("Continue with the most recent upload prepared for question answering.")
            if st.button("Continue Previous Upload", use_container_width=True):
                reset_chat()
                go_to("chat")
        else:
            st.write("No previous question-answering upload is available yet.")
            st.button("Continue Previous Upload", disabled=True, use_container_width=True)


def render_upload():
    render_top_bar()
    st.title("📤 Upload New Book Pages")
    st.write(
        "Upload one or more page images. Then choose whether to create a Word document "
        "or prepare the pages for question answering."
    )

    uploaded_files = st.file_uploader(
        "Choose book-page images",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("Upload at least one image to continue.")
        return

    st.success(f"{len(uploaded_files)} image(s) selected.")
    with st.expander("View uploaded file names"):
        for number, uploaded_file in enumerate(uploaded_files, start=1):
            st.write(f"{number}. {uploaded_file.name}")

    st.markdown("### What do you want to do with these images?")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📄 Convert to Word")
        st.write(
            "Runs OCR, layout, table/figure handling and Word generation. "
            "The previous RAG knowledge base is not changed."
        )
        if st.button("Convert to Word", type="primary", use_container_width=True):
            st.session_state.uploaded_files = uploaded_files
            go_to("word_processing")

    with col2:
        st.markdown("#### 💬 Ask Questions")
        st.write(
            "Runs OCR + knowledge export + chunking + embeddings + FAISS. "
            "No Word document is created."
        )
        if st.button("Prepare for Questions", type="primary", use_container_width=True):
            st.session_state.uploaded_files = uploaded_files
            st.session_state.rag_prepared = False
            st.session_state.rag_result = None
            go_to("rag_processing")


def render_word_processing():
    render_top_bar()
    st.title("📄 Convert Uploaded Pages to Word")
    uploaded_files = st.session_state.uploaded_files

    if not uploaded_files:
        st.warning("No uploaded images are available.")
        return

    progress_bar = st.progress(0)
    status = st.empty()

    def update_progress(current, total, message):
        progress_bar.progress(min(max(current / total if total else 0, 0.0), 1.0))
        status.write(message)

    try:
        with st.spinner("Processing pages..."):
            output_path = convert_uploads_to_word(
                uploaded_files,
                progress_callback=update_progress,
            )
        progress_bar.progress(1.0)
        status.success("Word document created successfully.")
        st.session_state.word_output_path = str(output_path)
    except Exception as error:
        st.error(f"Word conversion failed: {error}")
        return

    output_path = Path(st.session_state.word_output_path)
    if output_path.exists():
        st.success("✅ OCR completed\n\n✅ Layout processed\n\n✅ Word document generated")
        st.download_button(
            label="⬇️ Download Word Document",
            data=output_path.read_bytes(),
            file_name="converted_book.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True,
        )

    st.info("Your previous question-answering knowledge base was not changed.")


def render_rag_processing():
    render_top_bar()
    st.title("🧠 Prepare Uploaded Pages for Questions")

    uploaded_files = st.session_state.uploaded_files

    if not uploaded_files:
        st.warning("No uploaded images are available.")
        return

    # -----------------------------------------------------
    # IMPORTANT:
    # Streamlit reruns the full script every time a button is clicked.
    # Therefore, after RAG processing finishes we store the result
    # in session_state. On the next rerun, this block shows the
    # completed screen instead of going back to the processing button.
    # -----------------------------------------------------
    if (
        st.session_state.rag_prepared
        and st.session_state.rag_result
        and previous_rag_exists()
    ):
        result = st.session_state.rag_result

        st.success(
            "✅ OCR completed\n\n"
            "✅ Knowledge records created\n\n"
            "✅ Chunks created\n\n"
            "✅ Embeddings + FAISS index created"
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("Pages", result["pages"])
        col2.metric("OCR records", result["ocr_records"])
        col3.metric("Chunks", result["chunks"])

        st.info(
            "The uploaded pages are ready for question answering. "
            "No Word document was created."
        )

        if st.button(
            "💬 Start Asking Questions",
            type="primary",
            use_container_width=True,
        ):
            reset_chat()
            go_to("chat")

        return

    st.warning(
        "This will replace the previous question-answering upload. "
        "It will not create a Word document."
    )

    if not st.button(
        "Start OCR + RAG Processing",
        type="primary",
        use_container_width=True,
    ):
        return

    progress_bar = st.progress(0)
    status = st.empty()

    def update_progress(current, total, message):
        progress_bar.progress(
            min(
                max(
                    current / total if total else 0,
                    0.0,
                ),
                1.0,
            )
        )
        status.write(message)

    try:
        result = prepare_uploads_for_questions(
            uploaded_files,
            progress_callback=update_progress,
        )

        # Force the next chat load to use the newly created index.
        load_rag_resources.clear()
        reset_chat()

        # Persist successful processing across Streamlit reruns.
        st.session_state.rag_prepared = True
        st.session_state.rag_result = result

        progress_bar.progress(1.0)
        status.success(
            "Question-answering system is ready."
        )

        # Rerun intentionally. The session-state block at the top of
        # this function will now display the completed result screen.
        st.rerun()

    except Exception as error:
        st.session_state.rag_prepared = False
        st.session_state.rag_result = None
        st.error(
            f"Question-answering preparation failed: {error}"
        )

def render_chat():
    render_top_bar()
    st.title("💬 Ask Your Book")

    if not previous_rag_exists():
        st.warning("No processed question-answering upload was found.")
        return

    manifest_path = VECTOR_INDEX_DIRECTORY / "manifest.json"
    try:
        manifest_stamp = manifest_path.stat().st_mtime_ns
        settings, index, metadata, embedding_model, manifest, client = load_rag_resources(
            manifest_stamp
        )
        st.caption(f"Ready • {index.ntotal} searchable chunks • {settings['model_name']}")
    except Exception as error:
        st.error(f"Could not load the previous knowledge base: {error}")
        return

    _, clear_col = st.columns([5, 1])
    with clear_col:
        if st.button("Clear Chat", use_container_width=True):
            reset_chat()
            st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("sources"):
                with st.expander("📄 Sources"):
                    for source in message["sources"]:
                        st.write(f"• {source['image']} — Page {source['page_number']}")

    question = st.chat_input("Ask a question about the last processed upload...")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the scanned pages..."):
            try:
                result = answer_question(question)
                answer = result["answer"]
                sources = result["sources"]
            except Exception as error:
                answer = f"The question-answering request failed: {error}"
                sources = []

        st.markdown(answer)
        if sources:
            with st.expander("📄 Sources", expanded=True):
                for source in sources:
                    st.write(f"• {source['image']} — Page {source['page_number']}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })


screen = st.session_state.screen
if screen == "home":
    render_home()
elif screen == "upload":
    render_upload()
elif screen == "word_processing":
    render_word_processing()
elif screen == "rag_processing":
    render_rag_processing()
elif screen == "chat":
    render_chat()
else:
    st.session_state.screen = "home"
    st.rerun()
