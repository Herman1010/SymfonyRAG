import os
import json

from src import config
from src.ingest import load_rsts, prepare_docs, download_symfony_docs


def chunk_fixed(text: str, max_words: int = 500, overlap: int = 100):
    words = text.split()
    chunks = []
    step = max_words - overlap

    for i in range(0, len(words), step):
        chunk_words = words[i: i + max_words]
        chunk_text = " ".join(chunk_words)
        chunks.append(chunk_text)

    return chunks


def chunk_semantic(text: str, max_words: int = 500):
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks = []
    current_paras = []
    current_word_count = 0

    for para in paragraphs:
        words = para.split()
        nb_words = len(words)

        if current_paras and current_word_count + nb_words > max_words:
            chunks.append(" ".join(current_paras))
            current_paras = []
            current_word_count = 0

        current_paras.append(para)
        current_word_count += nb_words

    if current_paras:
        chunks.append(" ".join(current_paras))

    return chunks


def save_jsonl(chunks, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")


def load_chunks_jsonl(path):
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def build_all_chunks(docs, max_words: int = 500, overlap: int = 100):
    all_chunks_fixed = []
    all_chunks_semantic = []

    for doc in docs:
        # Fixed
        chunks_f = chunk_fixed(doc["clean_text"], max_words=max_words, overlap=overlap)
        for i, chunk_text in enumerate(chunks_f):
            all_chunks_fixed.append({
                "chunk_id": f"{doc['id']}_fixed_{i}",
                "source": doc["id"],
                "text": chunk_text,
                "title": doc["metadata"]["title"],
                "category": doc["metadata"]["category"],
            })

        # Semantic
        chunks_s = chunk_semantic(doc["clean_text"], max_words=max_words)
        for i, chunk_text in enumerate(chunks_s):
            all_chunks_semantic.append({
                "chunk_id": f"{doc['id']}_sem_{i}",
                "source": doc["id"],
                "text": chunk_text,
                "title": doc["metadata"]["title"],
                "category": doc["metadata"]["category"],
            })

    return all_chunks_fixed, all_chunks_semantic


def build_and_save_chunks(max_words: int = 500, overlap: int = 100):
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)

    docs = load_rsts()
    docs = prepare_docs(docs)

    all_fixed, all_sem = build_all_chunks(docs, max_words=max_words, overlap=overlap)

    fixed_path = os.path.join(config.PROCESSED_DIR, "chunks_fixed.jsonl")
    sem_path = os.path.join(config.PROCESSED_DIR, "chunks_semantic.jsonl")

    save_jsonl(all_fixed, fixed_path)
    save_jsonl(all_sem, sem_path)

    return {
        "fixed_path": fixed_path,
        "semantic_path": sem_path,
        "n_fixed": len(all_fixed),
        "n_semantic": len(all_sem),
    }


if __name__ == "__main__":
    
    download_symfony_docs()
    info = build_and_save_chunks(max_words=500, overlap=100)

    print("Chunks FIXED :", info["n_fixed"])
    print("Chunks SEMANTIC :", info["n_semantic"])
    print("Saved:", info["fixed_path"])
    print("Saved:", info["semantic_path"])
