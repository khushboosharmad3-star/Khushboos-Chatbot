# Khushboo's Chatbot

A local document-question-answering chatbot built with Streamlit, Sentence
Transformers, ChromaDB, and Ollama.

## Requirements

- macOS, Windows, or Linux
- Python 3.10 or newer
- Ollama installed and running
- Internet access for the first-time Python package and embedding-model downloads

## Setup on macOS (VS Code)

Open this folder in VS Code, then open **Terminal → New Terminal**.

### 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Python packages

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Install Ollama and download the local model

Download Ollama from https://ollama.com/download and install it.

In a terminal, run:

```bash
ollama pull llama3.2:3b
```

Keep the Ollama application running. If the Ollama server is not running,
you can start it in a separate terminal with:

```bash
ollama serve
```

If it says the port is already in use, Ollama is probably already running.

### 4. Start the chatbot

In the VS Code terminal, run:

```bash
streamlit run app.py
```

Open the local address shown in the terminal, usually:

http://localhost:8501

## How to use

1. Upload a PDF or TXT file in the sidebar.
2. Use a PDF with at least two pages. The app requires selectable text;
   scanned image-only PDFs need OCR first.
3. For TXT files, provide at least about 1,500 characters.
4. Click **Process document** and wait for text extraction and embeddings.
5. Ask questions in the chat box.
6. Expand **View source text** to inspect the retrieved passages.

## Project files

- `app.py` — Streamlit interface, extraction, chunking, embeddings, retrieval,
  Ollama answer generation, and source display.
- `requirements.txt` — Python dependencies.
- `README.md` — setup instructions.
- `chroma_db/` — created automatically to store the local vector database.

## Important notes

- The first run downloads the Sentence Transformers embedding model.
- Ollama runs the language model locally on your computer.
- The chatbot is designed to answer from retrieved document passages and to
  state when the information is unavailable. Retrieval thresholds may need
  adjustment for different documents; always verify results against the source.
- This starter project is intended for local use. Do not upload confidential
  documents to a public deployment without reviewing privacy and hosting needs.
