import os
import json
import faiss
from sentence_transformers import SentenceTransformer
from src import config
from src.chunking import load_chunks_jsonl, build_and_save_chunks  


# Modèle embeddings
EMB_MODEL = SentenceTransformer(config.MODEL_NAME)


def build_index(chunks, index_path, meta_path):
    texts = [c["text"] for c in chunks]
    print(f"Encodage de {len(texts)} chunks.")

    embeddings = EMB_MODEL.encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)

    with open(meta_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print("Index + meta sauvegardés :", index_path, meta_path)


def build_all_indexes():
    os.makedirs(config.INDEX_DIR, exist_ok=True)
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)

    fixed_chunks_path = os.path.join(config.PROCESSED_DIR, "chunks_fixed.jsonl")
    semantic_chunks_path = os.path.join(config.PROCESSED_DIR, "chunks_semantic.jsonl")

    # Si les chunks n'existent pas, on les génère
    if not os.path.exists(fixed_chunks_path) or not os.path.exists(semantic_chunks_path):
        print("Chunks manquants alors génération.")
        build_and_save_chunks(max_words=500, overlap=100)

    chunks_fixed = load_chunks_jsonl(fixed_chunks_path)
    chunks_semantic = load_chunks_jsonl(semantic_chunks_path)

    build_index(
        chunks_fixed,
        os.path.join(config.INDEX_DIR, "index_fixed.faiss"),
        os.path.join(config.INDEX_DIR, "meta_fixed.jsonl")
    )

    build_index(
        chunks_semantic,
        os.path.join(config.INDEX_DIR, "index_semantic.faiss"),
        os.path.join(config.INDEX_DIR, "meta_semantic.jsonl")
    )

    print("Index FIXED + SEMANTIC construits.")
    print("Contenu de index/ :", os.listdir(config.INDEX_DIR))

def load_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items


def load_index_and_meta(index_path, meta_path):
    index = faiss.read_index(index_path)
    metas = load_jsonl(meta_path)
    return index, metas

if __name__ == "__main__":
    build_all_indexes()
