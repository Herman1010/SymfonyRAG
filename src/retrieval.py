import os
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from src.parent_child import expand_with_neighbors
from src import config
from src.index_faiss import load_index_and_meta



# Chargement modèles & index

EMB_MODEL = SentenceTransformer(config.MODEL_NAME)

INDEX_FIXED_PATH = os.path.join(config.INDEX_DIR, "index_fixed.faiss")
META_FIXED_PATH  = os.path.join(config.INDEX_DIR, "meta_fixed.jsonl")

INDEX_SEM_PATH = os.path.join(config.INDEX_DIR, "index_semantic.faiss")
META_SEM_PATH  = os.path.join(config.INDEX_DIR, "meta_semantic.jsonl")

index_fixed, metas_fixed = load_index_and_meta(INDEX_FIXED_PATH, META_FIXED_PATH)
index_semantic, metas_semantic = load_index_and_meta(INDEX_SEM_PATH, META_SEM_PATH)



#  BM25 (sur fixed)

corpus_fixed = [m["text"] for m in metas_fixed]
tokenized_corpus_fixed = [doc.split() for doc in corpus_fixed]
bm25_fixed = BM25Okapi(tokenized_corpus_fixed)



#  Dense retrieval FAISS

def retrieve_dense(question: str, k: int = 5, mode: str = "fixed"):
    """
    mode = "fixed" ou "semantic"
    """
    if mode == "fixed":
        index = index_fixed
        metas = metas_fixed
    elif mode == "semantic":
        index = index_semantic
        metas = metas_semantic
    else:
        raise ValueError("mode doit être 'fixed' ou 'semantic'")

    q_emb = EMB_MODEL.encode([question], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)

    scores, indices = index.search(q_emb, k)
    scores = scores[0]
    indices = indices[0]

    results = []
    for rank, (idx, score) in enumerate(zip(indices, scores)):
        meta = metas[idx]
        results.append({
            "rank": rank,
            "score": float(score),
            "chunk_id": meta.get("chunk_id"),
            "text": meta["text"],
            "source": meta.get("source"),
            "title": meta.get("title"),
            "category": meta.get("category"),
        })
    return results



# BM25 retrieval (fixed)

def retrieve_bm25(question: str, k: int = 5):
    tokens_q = question.split()
    scores = bm25_fixed.get_scores(tokens_q)
    top_idx = np.argsort(scores)[::-1][:k]

    results = []
    for rank, idx in enumerate(top_idx):
        meta = metas_fixed[idx]
        results.append({
            "rank": rank,
            "bm25_score": float(scores[idx]),
            "chunk_id": meta.get("chunk_id"),
            "text": meta["text"],
            "source": meta.get("source"),
            "title": meta.get("title"),
            "category": meta.get("category"),
        })
    return results



# Hybrid dense + BM25 (fixed)

def retrieve_hybrid(question: str, k: int = 5, k_dense: int = 20, k_bm25: int = 20, alpha: float = 0.7):
    """
    Fusion  :
    - on prend top k_dense en dense
    - on prend top k_bm25 en bm25
    - on normalise scores
    - score_final = alpha*dense + (1-alpha)*bm25
    """
    dense = retrieve_dense(question, k=k_dense, mode="fixed")

    tokens_q = question.split()
    bm25_scores_all = bm25_fixed.get_scores(tokens_q)
    top_idx_bm25 = np.argsort(bm25_scores_all)[::-1][:k_bm25]

    # maps chunk_id -> score
    dense_map = {r["chunk_id"]: r["score"] for r in dense if r.get("chunk_id") is not None}
    bm25_map = {metas_fixed[i]["chunk_id"]: float(bm25_scores_all[i]) for i in top_idx_bm25 if metas_fixed[i].get("chunk_id") is not None}

    all_ids = set(dense_map.keys()) | set(bm25_map.keys())

    # retrouver meta par chunk_id 
    meta_by_id = {m.get("chunk_id"): m for m in metas_fixed}

    candidates = []
    for cid in all_ids:
        meta = meta_by_id.get(cid)
        if not meta:
            continue
        candidates.append({
            "chunk_id": cid,
            "text": meta["text"],
            "source": meta.get("source"),
            "title": meta.get("title"),
            "category": meta.get("category"),
            "dense_score": dense_map.get(cid, 0.0),
            "bm25_score": bm25_map.get(cid, 0.0),
        })

    if not candidates:
        return []

    dense_vals = np.array([c["dense_score"] for c in candidates], dtype=float)
    bm25_vals  = np.array([c["bm25_score"] for c in candidates], dtype=float)

    def minmax(x):
        if x.max() == x.min():
            return np.zeros_like(x)
        return (x - x.min()) / (x.max() - x.min())

    dense_norm = minmax(dense_vals)
    bm25_norm  = minmax(bm25_vals)

    for i, c in enumerate(candidates):
        c["dense_norm"] = float(dense_norm[i])
        c["bm25_norm"] = float(bm25_norm[i])
        c["score"] = alpha * c["dense_norm"] + (1 - alpha) * c["bm25_norm"]

    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)[:k]
    for rank, c in enumerate(candidates):
        c["rank"] = rank

    return candidates




def retrieve(question: str, k: int = 5, strategy: str = "hybrid", window: int = 1):
    if strategy == "fixed":
        return retrieve_dense(question, k=k, mode="fixed")
    if strategy == "semantic":
        return retrieve_dense(question, k=k, mode="semantic")
    if strategy == "bm25":
        return retrieve_bm25(question, k=k)
    if strategy == "hybrid":
        return retrieve_hybrid(question, k=k)

    if strategy == "parent_child":
        # child retrieval (simple) hybrid on fixed
        results = retrieve_hybrid(question, k=k)

        # expand context with neighbors (fixed)
        expanded = [expand_with_neighbors(r, metas_fixed, window) for r in results]
        return expanded

    raise ValueError(f"Strategy inconnue: {strategy}")




# Test

if __name__ == "__main__":
    q = "Comment définir une route simple dans Symfony ?"
    print("=== RETRIEVE HYBRID ===")
    res = retrieve(q, k=5, strategy="hybrid")
    for r in res:
        print(f"{r['rank']} | {r.get('source')} | score={r.get('score', r.get('bm25_score'))}")
