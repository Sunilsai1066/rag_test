# rag_index.py
import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

DOCS_DIR = "./docs"
PERSIST_DIR = "./chroma_db"
COLLECTION = "project_docs"


def load_all_pdfs(docs_dir: str):
    docs = []
    for name in sorted(os.listdir(docs_dir)):
        if name.lower().endswith(".pdf"):
            path = os.path.join(docs_dir, name)
            loader = PyPDFLoader(path)
            pdf_docs = loader.load()  # one Document per page with metadata: source, page
            docs.extend(pdf_docs)
            print(f"Loaded: {name} ({len(pdf_docs)} pages)")
    return docs


def main():
    Path(PERSIST_DIR).mkdir(parents=True, exist_ok=True)

    print("Loading PDFs...")
    raw_docs = load_all_pdfs(DOCS_DIR)

    print("Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      # adjust if your pages are very dense/sparse
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""],
    )
    docs = splitter.split_documents(raw_docs)
    print(f"Total chunks: {len(docs)}")

    print("Creating embeddings (local SentenceTransformer)...")
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("Building Chroma index (persisted to disk)...")
    db = Chroma.from_documents(
        documents=docs,
        embedding=embedding,
        persist_directory=PERSIST_DIR,
        collection_name=COLLECTION,
    )
    db.persist()
    print(f"Persisted to: {PERSIST_DIR}")


if __name__ == "__main__":
    main()
