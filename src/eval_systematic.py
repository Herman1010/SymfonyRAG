import json
import os
from dataclasses import dataclass
from typing import Dict, List

import sacrebleu
from rouge_score import rouge_scorer

from src.rag import ask_baseline, ask_rag, ask_rag_iterative


# ----------------------------
# Jeu de test (Q + référence)
# ----------------------------
# on écrit les "reference answers"  (courtes, 2-6 lignes),
# en se basant sur les docs Symfony de ton corpus.
TEST_SET_QA = [
    {
        "id": "routing_1",
        "question": "Comment définir une route simple dans Symfony ?",
        "reference": (
            "On peut définir une route via des attributs/annotations sur un contrôleur, "
            "ou via un fichier de configuration (YAML/XML/PHP). Exemple: #[Route('/', name: 'homepage')]."
        ),
        "expected_sources": ["routing.rst"],
    },
    {
        "id": "security_1",
        "question": "Comment sécuriser une page avec Symfony Security ?",
        "reference": (
            "On sécurise une page en configurant security.yaml (firewalls, access_control) "
            "et/ou en utilisant des attributs comme #[IsGranted('ROLE_ADMIN')] sur un contrôleur. "
            "Les rôles déterminent l'accès."
        ),
        "expected_sources": ["security.rst"],
    },
    {
        "id": "validation_1",
        "question": "Comment valider un formulaire avec Validator ?",
        "reference": (
            "On valide des données avec le Validator en utilisant des contraintes (Assert\\NotBlank, Assert\\Email, etc.) "
            "définies sur les propriétés (attributs/annotations) ou via YAML/XML, puis en appelant le validateur."
        ),
        "expected_sources": ["validation.rst"],
    },
]


# ----------------------------
#  Metrics: BLEU + ROUGE-L
# ----------------------------
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def compute_metrics(pred: str, ref: str) -> Dict[str, float]:
    pred = (pred or "").strip()
    ref = (ref or "").strip()

    # ROUGE-L F1
    rouge = scorer.score(ref, pred)["rougeL"].fmeasure

    # BLEU (sacrebleu)
    bleu = sacrebleu.sentence_bleu(pred, [ref]).score  # 0..100

    return {
        "rougeL_f1": float(rouge),
        "bleu": float(bleu),
    }


# ----------------------------
# 3) Génération (baseline / rag / iterative)
# ----------------------------
def generate_answers(q: str) -> Dict[str, str]:
    baseline = ask_baseline(q)

    rag_out = ask_rag(
        q,
        k=5,
        strategy="parent_child",   # on peut tester "hybrid" aussi
        use_rerank=True,
        window=1
    )
    rag = rag_out["answer"]

    it_out = ask_rag_iterative(
        q,
        k_final=5,
        strategy="parent_child",
        window=1
    )
    iterative = it_out["answer"]

    return {
        "baseline": baseline,
        "rag": rag,
        "iterative": iterative,
    }


# ----------------------------
# Eval globale + rapport
# ----------------------------
def evaluate_all(save_path: str = "eval_report.json", top_failures: int = 3):
    results: List[Dict] = []

    for item in TEST_SET_QA:
        qid = item["id"]
        q = item["question"]
        ref = item["reference"]

        answers = generate_answers(q)

        row = {
            "id": qid,
            "question": q,
            "reference": ref,
            "answers": answers,
            "metrics": {},
        }

        # metrics par système
        for sys_name, pred in answers.items():
            row["metrics"][sys_name] = compute_metrics(pred, ref)

        results.append(row)

    # moyennes
    def avg(metric_name: str, sys_name: str) -> float:
        vals = [r["metrics"][sys_name][metric_name] for r in results]
        return float(sum(vals) / len(vals)) if vals else 0.0

    summary = {
        "n": len(results),
        "avg": {
            "baseline": {
                "rougeL_f1": avg("rougeL_f1", "baseline"),
                "bleu": avg("bleu", "baseline"),
            },
            "rag": {
                "rougeL_f1": avg("rougeL_f1", "rag"),
                "bleu": avg("bleu", "rag"),
            },
            "iterative": {
                "rougeL_f1": avg("rougeL_f1", "iterative"),
                "bleu": avg("bleu", "iterative"),
            },
        },
    }

    # pire cas (on prend par défaut RAG itératif, ROUGE-L)
    sorted_fail = sorted(results, key=lambda r: r["metrics"]["iterative"]["rougeL_f1"])
    failures = sorted_fail[:top_failures]

    report = {
        "summary": summary,
        "failures_iterative_by_rougeL": [
            {
                "id": f["id"],
                "question": f["question"],
                "rougeL_f1": f["metrics"]["iterative"]["rougeL_f1"],
                "bleu": f["metrics"]["iterative"]["bleu"],
            }
            for f in failures
        ],
        "results": results,
    }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=== SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    print("\n=== TOP FAILURES (iterative, lowest ROUGE-L) ===")
    for f in report["failures_iterative_by_rougeL"]:
        print(f"- {f['id']} | rougeL={f['rougeL_f1']:.3f} | bleu={f['bleu']:.1f} | {f['question']}")

    print(f"\nReport saved to: {save_path}")


if __name__ == "__main__":
    evaluate_all()
