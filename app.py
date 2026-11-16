import hashlib
from io import BytesIO

import chromadb
import requests
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

st.set_page_config(
    page_title="Khushboo's Chatbot",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Khushboo's Chatbot")
st.caption("Your personal AI document assistant")
st.write(
    "Upload a PDF or TXT document, process it, and ask questions "
    "that are answered using the document's content."
)

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.2:3b"
DB_PATH = "./chroma_db"


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource
def get_database():
    return chromadb.PersistentClient(path=DB_PATH)


def extract_text(uploaded_file):
    filename = uploaded_file.name.lower()

    if filename.endswith(".pdf"):
        reader = PdfReader(BytesIO(uploaded_file.getvalue()))
        if len(reader.pages) < 2:
            raise ValueError("Please upload a PDF containing at least two pages.")

        pages = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append((text, f"Page {page_number}"))

        if not pages:
            raise ValueError(
                "No selectable text was found. If this is a scanned PDF, "
                "run OCR on it first and upload the searchable PDF."
            )
        return pages

    if filename.endswith(".txt"):
        text = uploaded_file.getvalue().decode("utf-8-sig", errors="replace")
        if len(text.strip()) < 1500:
            raise ValueError(
                "Please upload a TXT document with at least about two pages "
                "of information (roughly 1,500 characters or more)."
            )
        return [(text, "TXT document")]

    raise ValueError("Only PDF and TXT files are supported.")


def create_chunks(pages, chunk_size=900, overlap=150):
    chunks, sources = [], []

    for page_text, page_label in pages:
        start = 0
        while start < len(page_text):
            end = min(start + chunk_size, len(page_text))
            chunk = page_text[start:end].strip()

            if chunk:
                chunks.append(chunk)
                sources.append(page_label)

            if end >= len(page_text):
                break
            start = end - overlap

    return chunks, sources


def process_document(uploaded_file):
    pages = extract_text(uploaded_file)
    chunks, sources = create_chunks(pages)

    if not chunks:
        raise ValueError("No readable text was found in the document.")

    model = load_embedding_model()
    embeddings = model.encode(chunks, convert_to_numpy=True)

    collection_name = "doc_" + hashlib.md5(
        uploaded_file.getvalue()
    ).hexdigest()

    client = get_database()

    # Recreate this document's collection to avoid duplicate chunk IDs
    # when the same file is processed again.
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings.tolist(),
        metadatas=[{"source": sources[i]} for i in range(len(chunks))],
    )

    return collection, len(chunks)


def retrieve_information(question, collection):
    model = load_embedding_model()
    query_embedding = model.encode([question], convert_to_numpy=True)

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=min(4, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    documents = results["documents"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]

    relevant = []
    for document, distance, metadata in zip(documents, distances, metadatas):
        # This is a simple similarity cutoff; adjust after testing your documents.
        if distance <= 0.65:
            relevant.append(
                {
                    "text": document,
                    "source": metadata.get("source", "Unknown source"),
                    "distance": distance,
                }
            )
    return relevant


def generate_answer(question, context):
    prompt = f"""
You are Khushboo's Chatbot. Answer using ONLY the DOCUMENT CONTEXT below.

Rules:
- Do not use outside knowledge.
- Do not invent facts or fill gaps with assumptions.
- If the answer is not supported by the context, reply exactly:
  "This information is not available in the uploaded document."
- Give a clear, concise answer.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0},
        },
        timeout=180,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


with st.sidebar:
    st.header("📁 Upload document")
    uploaded_file = st.file_uploader(
        "Choose a PDF or TXT file",
        type=["pdf", "txt"],
    )

    if st.button("Process document", use_container_width=True):
        if uploaded_file is None:
            st.warning("Please upload a document first.")
        else:
            try:
                with st.spinner("Extracting text and creating embeddings..."):
                    collection, chunk_count = process_document(uploaded_file)
                    st.session_state.collection = collection
                    st.session_state.filename = uploaded_file.name
                    st.session_state.messages = []

                st.success("Document processed successfully!")
                st.info(f"Created {chunk_count} text chunks.")
            except Exception as error:
                st.error(str(error))

    st.divider()
    st.markdown("**Project:** Khushboo's Chatbot")
    st.markdown("**Vector database:** ChromaDB")
    st.markdown("**Embedding model:** all-MiniLM-L6-v2")
    st.markdown(f"**Local AI model:** {OLLAMA_MODEL}")


if "messages" not in st.session_state:
    st.session_state.messages = []
if "collection" not in st.session_state:
    st.session_state.collection = None

if st.session_state.collection is None:
    st.info("Upload a document in the sidebar and click **Process document** to begin.")
else:
    st.success(f"Currently reading: {st.session_state.filename}")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("📚 View source text"):
                    for source in message["sources"]:
                        st.markdown(f"**Source: {source['source']}**")
                        st.write(source["text"])

    question = st.chat_input("Ask a question about your document...")
    if question:
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching your document..."):
                try:
                    relevant_chunks = retrieve_information(
                        question, st.session_state.collection
                    )

                    if not relevant_chunks:
                        answer = (
                            "This information is not available in the uploaded document."
                        )
                        sources = []
                    else:
                        context = "\n\n".join(
                            item["text"] for item in relevant_chunks
                        )
                        answer = generate_answer(question, context)
                        sources = relevant_chunks

                    st.markdown(answer)

                    if sources:
                        with st.expander("📚 View source text used for this answer"):
                            for source in sources:
                                st.markdown(f"**{source['source']}**")
                                st.write(source["text"])

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                        }
                    )

                except requests.exceptions.ConnectionError:
                    st.error(
                        "Cannot connect to Ollama. Open the Ollama app, "
                        "make sure the model is installed, and try again."
                    )
                except requests.exceptions.HTTPError as error:
                    st.error(
                        f"Ollama returned an error: {error}. "
                        f"Check that `{OLLAMA_MODEL}` is installed."
                    )
                except Exception as error:
                    st.error(f"An error occurred: {error}")
