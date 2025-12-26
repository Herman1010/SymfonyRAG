import os
from dotenv import load_dotenv
from groq import Groq

from src.retrieval import retrieve
from src.rerank import rerank_with_cross_encoder


load_dotenv()  # lit .env (GROQ_API_KEY)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def call_llm(messages, model="llama-3.1-8b-instant", temperature=0.2, max_tokens=500):
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return resp.choices[0].message.content


def ask_baseline(question: str):
    messages = [
        {"role": "system", "content": "Tu es un assistant expert Symfony. Réponds clairement."},
        {"role": "user", "content": question},
    ]
    return call_llm(messages)


def build_rag_prompt(question: str, chunks: list):
    context = ""
    for i, ch in enumerate(chunks, start=1):
        src = ch.get("source", "unknown")
        context += f"[Source {i} - {src}]\n{ch['text']}\n\n"

    prompt = (
        "Tu es un assistant expert Symfony.\n"
        "Réponds à la question en te basant uniquement sur le CONTEXTE ci-dessous.\n"
        "Si l'information n'est pas dans le contexte, dis-le.\n\n"
        f"CONTEXTE:\n{context}\n"
        f"QUESTION:\n{question}\n\n"
        "RÉPONSE:"
    )
    return prompt


def ask_rag(question: str, k: int = 5, strategy: str = "hybrid", use_rerank: bool = True, window: int = 1):

    # Retrieval 
    if strategy == "parent_child":
        retrieved = retrieve(question, k=6, strategy="parent_child", window=window)
    else:
        retrieved = retrieve(question, k=10, strategy=strategy)

    # Rerank
    if use_rerank:
        top_chunks = rerank_with_cross_encoder(question, retrieved, k=k)
    else:
        top_chunks = retrieved[:k]

    # Sécurité taille
    MAX_CHARS_PER_CHUNK = 1500
    for c in top_chunks:
        if "text" in c and len(c["text"]) > MAX_CHARS_PER_CHUNK:
            c["text"] = c["text"][:MAX_CHARS_PER_CHUNK]

    # Prompt + LLM
    prompt = build_rag_prompt(question, top_chunks)

    messages = [
        {"role": "system", "content": "Tu es un assistant expert Symfony."},
        {"role": "user", "content": prompt},
    ]
    answer = call_llm(messages)

    return {
    "answer": answer,
    "chunks": top_chunks,
    "sources": [
        {
            "source": c.get("source"),
            "chunk_id": c.get("chunk_id"),
            "expanded_from": c.get("expanded_from"),
            "window": c.get("window"),
        }
        for c in top_chunks
    ],
}


    

def generate_alternative_queries(question: str, n: int = 3):
    prompt = (
        "Génère " + str(n) + " reformulations différentes  de la question suivante.\n"
        "Retourne uniquement une liste, une reformulation par ligne.\n\n"
        f"Question: {question}"
    )
    messages = [{"role": "user", "content": prompt}]
    text = call_llm(messages, temperature=0.7, max_tokens=200)

    lines = [l.strip("- ").strip() for l in text.split("\n") if l.strip()]
    return lines[:n]


def ask_rag_multi_query(question: str, k: int = 5, use_rerank: bool = True):
    #  reformulations
    queries = [question] + generate_alternative_queries(question, n=3)

    # retrieval sur chaque query
    all_hits = []
    seen = set()
    for q in queries:
        hits = retrieve(q, k=5, strategy="hybrid")
        for h in hits:
            cid = h.get("chunk_id")
            if cid and cid not in seen:
                seen.add(cid)
                all_hits.append(h)

    #  rerank 
    if use_rerank:
        top_chunks = rerank_with_cross_encoder(question, all_hits, k=k)
    else:
        top_chunks = all_hits[:k]

    prompt = build_rag_prompt(question, top_chunks)
    messages = [
        {"role": "system", "content": "Tu es un assistant expert Symfony."},
        {"role": "user", "content": prompt},
    ]
    answer = call_llm(messages)

    return {
        "answer": answer,
        "chunks": top_chunks, 
    }


if __name__ == "__main__":
    q = "Comment définir une route simple dans Symfony ?"

    print("=== BASELINE ===")
    print(ask_baseline(q))

    print("\n=== RAG (hybrid + rerank) ===")
    out = ask_rag(q, k=5, strategy="hybrid", use_rerank=True)
    print(out["answer"])
    print("Sources:", out["sources"])
def ask_rag_iterative(
    question: str,
    k_final: int = 5,
    strategy: str = "parent_child",
    window: int = 1,
    n_subqueries: int = 3,
):
    
    
    
    
    
    
    
    """
    RAG itératif (2 tours) :
    - Tour 1 : RAG standard
    - Tour 2 : génération de sous-questions -> retrieval -> réponse finale
    """

    # --------------------
    #  (RAG standard)
    # --------------------
    out1 = ask_rag(question, k=min(3, k_final), strategy=strategy, use_rerank=True, window=window)
    draft = out1["answer"]

    # --------------------
    # Générer des sous-questions (agent)
    # --------------------
    prompt_subq = (
        "Tu aides à faire une recherche documentaire dans la doc Symfony.\n"
        "À partir de la question et de la réponse provisoire, propose "
        f"{n_subqueries} sous-questions très courtes  "
        "pour compléter l'information.\n\n"
        f"Question: {question}\n\n"
        f"Réponse provisoire:\n{draft}\n"
    )

    subq_text = call_llm(
        [{"role": "user", "content": prompt_subq}],
        temperature=0.2,
        max_tokens=120
    )

    subqueries = [l.strip("-• ").strip() for l in subq_text.split("\n") if l.strip()]
    subqueries = subqueries[:n_subqueries]

    # --------------------
    # (retrieval sur sous-questions + fusion)
    # --------------------
    all_hits = []
    seen_ids = set()

    for sq in subqueries:
        hits = retrieve(sq, k=6, strategy=strategy, window=window) if strategy == "parent_child" \
               else retrieve(sq, k=10, strategy=strategy)

        for h in hits:
            cid = h.get("chunk_id")
            if cid and cid not in seen_ids:
                seen_ids.add(cid)
                all_hits.append(h)

    # Rerank final sur question originale
    top_chunks = rerank_with_cross_encoder(question, all_hits, k=k_final)

    #  tronquer
    MAX_CHARS_PER_CHUNK = 1500
    for c in top_chunks:
        if "text" in c and len(c["text"]) > MAX_CHARS_PER_CHUNK:
            c["text"] = c["text"][:MAX_CHARS_PER_CHUNK]

    # Prompt final + réponse finale
    final_prompt = build_rag_prompt(question, top_chunks)

    final_messages = [
        {"role": "system", "content": "Tu es un assistant expert Symfony. Réponds en te basant UNIQUEMENT sur le contexte fourni."},
        {"role": "user", "content": final_prompt},
    ]

    final_answer = call_llm(final_messages, temperature=0.2, max_tokens=450)

    return {
        "answer": final_answer,
        "subqueries": subqueries,
        "chunks": top_chunks,
        "sources": [
            {
                "source": c.get("source"),
                "chunk_id": c.get("chunk_id"),
                "expanded_from": c.get("expanded_from"),
                "window": c.get("window"),
            }
            for c in top_chunks
        ],
    }
    def format_sources(chunks, top: int = 5):
     out = []
    for c in chunks[:top]:
        out.append({
            "source": c.get("source"),
            "chunk_id": c.get("chunk_id"),
            "score": c.get("score", c.get("dense_score", c.get("bm25_score"))),
            "dense_norm": c.get("dense_norm"),
            "bm25_norm": c.get("bm25_norm"),
            "window": c.get("window"),
        })
    return out
