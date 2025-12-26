# Symfony Doc Assistant : Système RAG Intelligent

Ce projet implémente un système de **RAG (Retrieval-Augmented Generation)** permettant d'interroger la documentation officielle de Symfony 7.3. L'objectif est de fournir des réponses précises, contextualisées et sans hallucinations en s'appuyant sur les sources officielles du framework.

## Problématique

La documentation technique de Symfony est extrêmement vaste. Pour un développeur, trouver la syntaxe exacte ou une configuration spécifique peut prendre du temps.  
Le défi : créer un assistant qui ne se contente pas de "connaître" Symfony, mais qui va chercher l'information en temps réel dans les fichiers sources (.rst) pour garantir une réponse fiable et sourcée.

## Dataset

Le dataset est constitué des fichiers sources de la documentation officielle de Symfony version 7.3 (format ReStructuredText - RST).

- **Source** : [GitHub Symfony Docs](https://github.com/symfony/symfony-docs)  
- **Contenu** : Plusieurs fichiers clés incluant `routing.rst`, `security.rst`, `doctrine.rst`, `service_container.rst`, etc.  
- **Format** : Données brutes nettoyées et converties en texte simple pour le traitement.

## Solutions Appliquées & Technologies

### 1. Pipeline de Données
- **Extraction** : Récupération des fichiers directement depuis GitHub via `requests`.  
- **Nettoyage** : Suppression des directives RST, balises de mise en forme et normalisation du texte avec `re`.  
- **Chunking (Découpage)** :  
  - **Fixed Chunking** : Découpage par blocs de mots avec recouvrement (overlap).  
  - **Semantic Chunking** : Découpage basé sur la structure des paragraphes pour conserver la cohérence logique.

### 2. Indexation Vectorielle
- **Embeddings** : Modèle `all-MiniLM-L6-v2` via `sentence-transformers` pour transformer chaque bloc de texte en vecteur (dimension 384).  
- **Base de données vectorielle** : FAISS (Facebook AI Similarity Search) pour des recherches de similarité cosinus rapides.

### 3. Moteur de Recherche Hybride
- **Dense Retrieval** : Recherche vectorielle via FAISS pour capturer le sens sémantique.  
- **Sparse Retrieval** : Recherche par mots-clés avec BM25 via `rank_bm25`.

### 4. Optimisation de la Requête : Multi-Query Retrieval
- Génération automatique de plusieurs variantes de la question initiale via le LLM.  
- Objectif : explorer différentes zones de l’espace vectoriel pour capturer des documents sémantiquement proches mais lexicalement différents.  
- Résultat : augmentation du taux de rappel (Recall).

### 5. Stratégie de Recherche : Hybrid Search + Reranking
- **Recherche Hybride** : Combinaison de FAISS (sémantique) et BM25 (lexicale).  
- **Reranking** : Passage des documents récupérés dans un modèle Cross-Encoder pour réordonner selon la pertinence.

### 6. Génération (LLM)
- **Modèle** : `llama-3.1-8b-instant` via l'API Groq.  
- **Prompt Engineering** : Injection du contexte récupéré pour forcer le modèle à répondre uniquement selon les documents fournis ("Grounding").

## Résultats Obtenus

### 1. Analyse qualitative du Retrieval et du Reranking
- Chaque réponse générée est accompagnée des chunks documentaires utilisés comme contexte.  
- Pour chaque chunk : score FAISS, score BM25 et score de reranking (Cross-Encoder).  
- Résultat : récupération efficace des documents à la fois sémantiquement proches et lexicalement pertinents, avec un ordre optimisé pour la génération.

### 2. Évaluation du Retrieval par Precision@k et Recall@k
- **Métriques** :  
  - Precision@k = proportion de chunks pertinents parmi les k premiers.  
  - Recall@k = proportion de chunks pertinents retrouvés parmi tous les chunks pertinents attendus.  
- **Résultats** :  
  - Precision@k = 0.9  
  - Recall@k = 1.0  
- Conclusion : le retrieval hybride + reranking garantit une forte couverture et un faible taux de bruit.

### 3. Évaluation globale par ROUGE-L et BLEU
- Comparaison des réponses générées à des réponses de référence.  
- Métriques utilisées :  
  - **ROUGE-L (F1)** : mesure la similarité structurelle.  
  - **BLEU** : mesure la similarité lexicale via le chevauchement des n-grams.  
- Configurations testées : baseline sans retrieval, RAG standard, RAG itératif avec reformulation.  
- Résultat : les approches RAG surpassent le baseline, offrant des réponses plus structurées et fidèles à la documentation.

### 4. Discussion des résultats
- Le retrieval hybride assure une récupération efficace des informations pertinentes.  
- Le reranking améliore la qualité du contexte fourni au LLM.  
- L’intégration du RAG augmente significativement la qualité des réponses.  
- Les métriques automatiques ont certaines limites face aux reformulations ou aux réponses plus détaillées.

## Installation et Utilisation

### Pré-requis
- Python 3.9+  
- Clé API Groq

### Installation
Clônez le répo et à la racine créez un fichier .env et mettez cette ligne
```bash
GROQ_API_KEY="Ma clé GROQ"
```
```bash
pip install -r requirements.txt
```

### Démo
```bash
python -m src.main
python -m src.eval
python -m src.eval_systematic
```
