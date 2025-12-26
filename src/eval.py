from typing import List, Set
from src.retrieval import retrieve
from src.rag import ask_baseline, ask_rag

TEST_SET = [
    {"q": "Comment définir une route simple dans Symfony ?", "expected_sources": {"routing.rst"}},
    {"q": "Comment sécuriser une page avec Symfony Security ?", "expected_sources": {"security.rst"}},
    {"q": "Comment valider un formulaire avec Validator ?", "expected_sources": {"validation.rst"}},
    {"q": "Comment configurer un service dans le service container ?", "expected_sources": {"service_container.rst"}},
    {"q": "Comment utiliser Messenger pour traiter des messages ?", "expected_sources": {"messenger.rst"}},
    {"q": "Comment traduire des messages avec Translation ?", "expected_sources": {"translation.rst"}},
    {"q": "Comment sérialiser un objet avec Serializer ?", "expected_sources": {"serializer.rst"}},
    {"q": "Comment envoyer un email avec Mailer ?", "expected_sources": {"mailer.rst"}},
]

def precision_at_k(retrieved_sources: List[str], expected: Set[str], k: int) -> float:
    topk = retrieved_sources[:k]
    if not topk:
        return 0.0
    hits = sum(1 for s in topk if s in expected)
    return hits / len(topk)


def recall_at_k(retrieved_sources: List[str], expected: Set[str], k: int) -> float:
    topk = retrieved_sources[:k]
    return 1.0 if any(s in expected for s in topk) else 0.0


def eval_retrieval(k: int = 5, strategy: str = "hybrid"):
    precisions, recalls = [], []

    for item in TEST_SET:
        q = item["q"]
        expected = item["expected_sources"]

        hits = retrieve(q, k=k, strategy=strategy)
        sources = [h.get("source") for h in hits if h.get("source")]

        precisions.append(precision_at_k(sources, expected, k))
        recalls.append(recall_at_k(sources, expected, k))

    return {
        "k": k,
        "strategy": strategy,
        "Precision@k": sum(precisions) / len(precisions),
        "Recall@k": sum(recalls) / len(recalls),
        "n_questions": len(TEST_SET),
    }


def qualitative_compare(n: int = 5):
    """
    Compare baseline vs RAG sur n questions et affiche les sources.
    """
    for i, item in enumerate(TEST_SET[:n], start=1):
        q = item["q"]

        print("\n" + "=" * 80)
        print(f"Q{i}: {q}")
        print("=" * 80)

        print("\n--- BASELINE (sans RAG) ---")
        print(ask_baseline(q))

        print("\n--- RAG (hybrid + rerank) ---")
        out = ask_rag(q, k=5, strategy="hybrid", use_rerank=True)
        print(out["answer"])

        sources = [c.get("source") for c in out["chunks"]]
        print("Sources:", sources)


# Cas d'échec (Failure cases)


FAILURE_QUESTIONS = [
    # Hors corpus
    "Comment déployer Symfony sur Kubernetes avec Helm ?",

    # Trop vague
    "Explique Symfony en général.",

    # Trop large / irréaliste
    "Donne un exemple complet de microservices en Symfony."
]


def analyze_failures(k: int = 5, strategy: str = "hybrid"):
    """
    Analyse les échecs en distinguant :
    - problème de retrieval
    - problème de génération
    """
    for q in FAILURE_QUESTIONS:
        print("\n" + "=" * 80)
        print("FAIL CASE:", q)
        print("=" * 80)

        # RETRIEVAL 
        hits = retrieve(q, k=k, strategy=strategy)
        sources = [h.get("source") for h in hits if h.get("source")]

        print("\n[Retrieval]")
        print("Top sources:", sources)

        if not hits:
            print("Diagnostic: Aucun chunk récupéré (échec de retrieval).")
        elif len(set(sources)) == 1:
            print("Diagnostic: Résultats peu diversifiés (un seul document dominant).")
        else:
            print("Diagnostic: Retrieval partiellement pertinent.")

        #  GENERATION
        out = ask_rag(q, k=5, strategy=strategy, use_rerank=True)

        print("\n[Generation - RAG answer]")
        print(out["answer"])

        answer_lower = out["answer"].lower()

        if any(x in answer_lower for x in [
            "pas dans le contexte",
            "information non disponible",
            "je ne sais pas",
            "n'est pas disponible"
        ]):
            print("Diagnostic: Le modèle indique correctement l'absence d'information.")
        else:
            print("Diagnostic: Risque d'hallucination ou réponse trop générale.")


if __name__ == "__main__":
    print("=== Évaluation Retrieval ===")
    metrics = eval_retrieval(k=5, strategy="hybrid")
    print(metrics)

    print("\n=== Comparaison qualitative Baseline vs RAG ===")
    qualitative_compare(n=5)

    print("\n=== Analyse des cas d'échec ===")
    analyze_failures(k=5, strategy="hybrid")
