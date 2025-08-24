import argparse
import os
import warnings
from typing import List, Tuple

# Suppress warnings & logs
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # hides tensorflow logs if any
warnings.filterwarnings("ignore")  # hides all warnings
import torch

# Optional: suppress just LangChain deprecations
try:
    from langchain_core._api import LangChainDeprecationWarning

    warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
except ImportError:
    pass

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import pipeline

PERSIST_DIR = "./chroma_db"
COLLECTION = "project_docs"


def load_retriever(k: int = 4):
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma(
        embedding_function=embedding,
        persist_directory=PERSIST_DIR,
        collection_name=COLLECTION,
    )
    return db.as_retriever(search_kwargs={"k": k})


def build_local_llm():
    device = 0 if torch.cuda.is_available() else -1  # 0 = first GPU, -1 = CPU
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-large",
        device=device,
    )


def make_prompt(query: str, contexts: List[str]) -> str:
    context_block = "\n\n---\n\n".join(contexts)
    # This new prompt template instructs the model to filter the context first
    return (
        "You are a helpful assistant. Your task is to answer a question based on the context below.\n\n"
        "INSTRUCTIONS:\n"
        "1. First, carefully read the entire context to find the section that is most relevant to the user's question.\n"
        "2. Answer the question using ONLY the information from that single most relevant section.\n"
        "3. Be concise and do not include information from other, irrelevant sections. Do not repeat the question or section titles.\n\n"
        f"--- CONTEXT ---\n{context_block}\n\n"
        f"--- QUESTION ---\n{query}\n\nAnswer:"
    )


def answer_query(query: str, show_sources: bool = True, top_k: int = 4) -> Tuple[str, List[dict]]:
    retriever = load_retriever(k=top_k)
    docs = retriever.get_relevant_documents(query)
    contexts = [d.page_content for d in docs]

    llm = build_local_llm()
    prompt = make_prompt(query, contexts)
    out = llm(prompt)[0]["generated_text"].strip()

    sources = []
    if show_sources:
        for d in docs:
            meta = d.metadata or {}
            sources.append({
                "source": meta.get("source", "unknown"),
                "page": meta.get("page", "unknown"),
            })
    return out, sources


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ask questions to your local RAG system")
    parser.add_argument("question", type=str, help="The question to ask")
    parser.add_argument("--no-sources", action="store_true", help="Hide sources in output")
    parser.add_argument("--k", type=int, default=4, help="Number of chunks to retrieve")
    args = parser.parse_args()

    ans, srcs = answer_query(args.question, show_sources=not args.no_sources, top_k=args.k)

    print(f"\nQ: {args.question}\n")
    print(f"A: {ans}\n")
    if not args.no_sources and srcs:
        print("Sources:")
        for s in srcs:
            print(f" - {s['source']} (page {s['page']})")
