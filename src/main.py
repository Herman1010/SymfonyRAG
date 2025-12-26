from src.rag import ask_baseline, ask_rag, ask_rag_multi_query

QUESTIONS = [
    "Comment définir une route simple dans Symfony ?",
    "Comment sécuriser une page avec Symfony Security ?",
    "Comment valider un formulaire avec Validator ?",
    "Comment configurer un service dans le service container ?",
    "Comment utiliser Messenger pour traiter des messages ?",
    "Comment traduire des messages avec Translation ?",
    "Comment sérialiser un objet avec Serializer ?",
    "Comment envoyer un email avec Mailer ?",
    "Comment utiliser Doctrine pour mapper une entité ?",
    "Quelles bonnes pratiques de performance recommande Symfony ?",
]

def print_chunks_with_scores(chunks):
    for j, c in enumerate(chunks, start=1):
        print(
            f"  {j}. {c.get('source')} | {c.get('chunk_id')} | "
            f"score={c.get('score')} | bm25={c.get('bm25_score')} | rerank={c.get('rerank_score')}"
        )

def run_demo():
    for i, q in enumerate(QUESTIONS, start=1):
        print("\n" + "=" * 80)
        print(f"Q{i}: {q}")
        print("=" * 80)

        print("\n--- BASELINE (sans RAG) ---")
        baseline = ask_baseline(q)
        print(baseline)

        print("\n--- RAG (hybrid + rerank) ---")
        out = ask_rag(q, k=5, strategy="hybrid", use_rerank=True)
        print(out["answer"])
        print("Chunks & scores:")
        print_chunks_with_scores(out["chunks"])

        print("\n--- RAG MULTI-QUERY (hybrid + rerank) ---")
        out_mq = ask_rag_multi_query(q, k=5, use_rerank=True)
        print(out_mq["answer"])
        print("Chunks & scores (MQ):")
        print_chunks_with_scores(out_mq["chunks"])

if __name__ == "__main__":
    run_demo()
from src.rag import ask_rag_iterative
q = "Comment sécuriser une page avec Symfony Security ?"

out_it = ask_rag_iterative(q, k_final=5, strategy="parent_child", window=1)
print(out_it["answer"])
print("Subqueries:", out_it["subqueries"])
print("Sources IT:", out_it["sources"])
