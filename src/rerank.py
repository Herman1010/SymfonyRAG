from sentence_transformers import CrossEncoder
from src import config


# Modèle reranker (cross-encoder)
CROSS_ENCODER = CrossEncoder(config.CROSS_ENCODER_MODEL)


def rerank_with_cross_encoder(question: str, retrieved_chunks: list, k: int = 5):
    """
    retrieved_chunks: liste de dicts qui contiennent au moins "text"
    Retourne les top-k rerankés.
    """
    pairs = [(question, ch["text"]) for ch in retrieved_chunks]
    scores = CROSS_ENCODER.predict(pairs)

    # On ajoute un score rerank et on trie
    for ch, s in zip(retrieved_chunks, scores):
        ch["rerank_score"] = float(s)

    reranked = sorted(retrieved_chunks, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:k]


if __name__ == "__main__":
    from src.retrieval import retrieve

    q = "Comment définir une route simple dans Symfony ?"
    retrieved = retrieve(q, k=10, strategy="hybrid")

    top = rerank_with_cross_encoder(q, retrieved, k=5)

    print("RERANK TOP 5 ")
    for r in top:
        print(r["rank"], "|", r.get("source"), "| rerank_score=", r["rerank_score"])
