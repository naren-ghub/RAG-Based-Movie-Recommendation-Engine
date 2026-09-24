# 🎬 FilmDB: 3-Stage Hybrid RAG Movie Recommendation Engine

[![Tech Stack](<https://img.shields.io/badge/Stack-Python%20|%20LlamaIndex%20|%20Qdrant%20|%20Groq-blue>)](https://github.com/naren-ghub/FilmDB)
[![Kaggle Showcase](<https://img.shields.io/badge/Kaggle-Interactive%20Showcase-20BEFF?logo=kaggle&logoColor=white>)](https://www.kaggle.com/code/naren2308/filmdb-recommendations-engine-showcase)

A semantic discovery engine that understands the "soul" of cinema. Moving beyond simplistic collaborative filtering and naive keyword matching, **FilmDB** utilizes a **3-stage hybrid RAG pipeline** powered by **LlamaIndex** and **Qdrant Cloud** across a curated corpus of **95,000+ films**.

From *"mind-bending loops with existential dread"* to *"gritty 70s neo-noir set in rainy Tokyo,"* it delivers hyper-curated, diversity-aware recommendations with structured natural language justifications.

---

## 🚀 Explore the Showcase

Want to see how the engine handles queries like *"Recommend me a quiet, slice-of-life story about a family in Tokyo"*?

Explore the interactive evaluation pipeline live on Kaggle. **(No GPU or API keys required!)**

👉 **[CLICK HERE to Launch the Interactive Kaggle Showcase](https://www.kaggle.com/code/naren2308/filmdb-recommendations-engine-showcase)**

---

## ⚙️ The 3-Stage RAG Pipeline

Our architecture ensures ultra-fast candidate retrieval followed by deep contextual reranking and agentic reasoning.

```
User Query / Vibe
      │
      ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 1️⃣ STAGE 1: HYBRID RETRIEVAL & PRE-FILTERING (Qdrant Cloud)            │
│ • Dense Embeddings: BAAI/bge-base-en-v1.5 (768-d float32)              │
│ • Sparse Matching: BM25 Lexical Scoring                                │
│ • Rank Fusion: Reciprocal Rank Fusion (RRF, k=60)                     │
│ • Pre-Filtering: 14 payload filters (Streaming, Oscars, Year, Votes)   │
└────────────────────────────────────────────────────────────────────────┘
      │ Top 50 Candidates
      ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 2️⃣ STAGE 2: CROSS-ENCODER RERANKING & DIVERSITY OPTIMIZATION           │
│ • Deep Cross-Attention: BAAI/bge-reranker-large                        │
│ • Anti-Clustering: Maximal Marginal Relevance (MMR, λ=0.7)            │
│ • Hidden Gem Surfacing: 60/40 Gem-Score / Relevance Blending           │
└────────────────────────────────────────────────────────────────────────┘
      │ Top 10 High-Precision Matches
      ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 3️⃣ STAGE 3: AGENTIC LLM CURATION & JUSTIFICATION                      │
│ • Model: LLaMA-3.3-70B-Versatile (via Groq LPU)                        │
│ • Orchestration: LlamaIndex ResponseSynthesizer                        │
│ • Output: Top 5 selected films with evidence-backed explanations       │
└────────────────────────────────────────────────────────────────────────┘
```

### 1️⃣ Stage 1: Hybrid Retrieval & Pre-Filtering

* **Technology:** `BAAI/bge-base-en-v1.5` (768-d embeddings) + Sparse BM25 + **Qdrant Cloud**
* **Process:** Every movie is represented through a rich semantic template (Title, Year, Genres, Tone, Directors, Plot). When a user submits a query, candidate pools from dense vector search and sparse BM25 are merged dynamically using **Reciprocal Rank Fusion (RRF, $k=60$)**. Concurrently, Qdrant executes 14 payload filters (streaming services, minimum IMDb ratings, Oscar wins, languages) to pull the top 50 highly relevant candidates from the 95k corpus.

### 2️⃣ Stage 2: Cross-Encoder Reranking & Diversity Optimization

* **Technology:** `BAAI/bge-reranker-large` + Maximal Marginal Relevance (MMR)
* **Process:** Bi-encoders are fast at filtering, but lack deep cross-attention between queries and documents. The top 50 candidates are fed into a heavy Cross-Encoder to evaluate query-movie token interactions simultaneously. The reranked pool is then refined using:
  * **MMR (Maximal Marginal Relevance):** Penalizes inter-document redundancy on open-ended queries to prevent franchise repetition (e.g., returning 4 sequels of the same film).
  * **Hidden Gem Discovery:** Interpolates semantic relevance with an algorithmic `gem_score` (60/40 blend) to bubble up critically acclaimed, lesser-known indie masterpieces.
  * Filters down to the top 10 refined candidates.

### 3️⃣ Stage 3: LLM Curation & Explanation

* **Technology:** `llama-3.3-70b-versatile` (Running via Groq API) + LlamaIndex `ResponseSynthesizer`
* **Process:** The refined top 10 list is passed into LLaMA 3.3 70B running on ultra-low-latency Groq hardware. Acting as a senior cinematic curator, the LLM enforces a strict quality guard, selects the absolute best 5 films from the provided context, and generates a personalized, structured explanation of exactly why each film fits the user's specific emotional and stylistic request.

---

## 🧠 Agentic Orchestration

* **Technology:** **LlamaIndex**
* **Capability:** Serves as the backbone orchestrating data retrieval, metadata pre-filtering (streaming availability, Oscar count, release era), dynamic query routing, and real-time **Out-Of-Vocabulary (OOV)** handling for recent releases via TMDB API integration.

---

## 🛠 Project Lifecycle: Phase by Phase

This repository is divided into sequential Jupyter Notebooks, documenting the entire pipeline from raw data to the interactive showcase:

### Phase 1: Data Curation (`01_data_curation.ipynb`)

We started with a massive raw dataset of 347,000+ movies. We applied strict quality filters (deduplication, removing entries with 0 votes, filtering extreme outliers) to curate a high-quality, dense corpus of **95,090 films**. Metadata strings were cleaned, enriched, and normalized for the embedding pipeline.

### Phase 2: Embedding Generation & Indexing (`02_embedding_generation.ipynb` & `03_qdrant_indexing.ipynb`)

Each movie's metadata was injected into a highly engineered text template. Using `BAAI/bge-base-en-v1.5`, we generated 95,090 float32 vector embeddings. These vectors, along with rich payload metadata (e.g., `imdb_rating`, `director`, `oscar_wins`, `gem_score`, `streaming_platforms`), were indexed into a **Qdrant Cloud** cluster for sub-millisecond HNSW search.

### Phase 3: RAG Multi-Strategy Evaluation (`05_rag_evaluation.ipynb`)

To validate the pipeline, we designed a Retrieval Strategy Catalog consisting of **9 distinct strategies** (e.g., `S2_Prestige`, `S6_Director`, `S7_Diversity` with MMR, `S8_Mood`, `S9_HiddenGem`) evaluated across **21 highly specific semantic queries**:

1. Executed the complete evaluation pipeline on a Kaggle T4 GPU.
2. Logged outputs and metrics for Stage 1 (Hybrid RRF), Stage 2 (Cross-Encoder / MMR), and Stage 3 (LLM Curation).
3. Exported the final results as a static JSON artifact (`showcase_results.json`).

### Phase 4: The Interactive Showcase (`06_showcase.ipynb` / `filmdb-recommendations-engine-showcase.ipynb`)

Because the heavy GPU computation was pre-computed during Phase 3, we built an interactive dashboard that runs entirely on CPU. By loading the `showcase_results.json` dataset, the Showcase notebook uses `ipywidgets` and `plotly` to render an interactive UI. Users can explore Head-to-Head pipeline comparisons, latency metrics, and reranker quality lift charts without needing an API key or GPU.

---

## 🛠️ Production Tech Stack

| Component                  | Technology                                             | Role                                                               |
| -------------------------- | ------------------------------------------------------ | ------------------------------------------------------------------ |
| **Orchestration**    | [LlamaIndex](https://www.llamaindex.ai/)                | Index abstractions, retrievers, postprocessors, response synthesis |
| **Vector Database**  | [Qdrant Cloud](https://qdrant.tech/)                    | Managed HNSW vector store with 14 payload metadata filters         |
| **Embeddings**       | `BAAI/bge-base-en-v1.5`                              | 768-dimensional dense bi-encoder text representations              |
| **Reranker**         | `BAAI/bge-reranker-large`                            | Cross-encoder for query-document interaction scoring               |
| **Diversity Engine** | Maximal Marginal Relevance (MMR)                       | Anti-clustering algorithm to prevent franchise collapse            |
| **LLM Inference**    | [Groq](https://groq.com/) (`llama-3.3-70b-versatile`) | Ultra-fast token generation for cinematic justifications           |
| **Corpus & Data**    | Python / Polars / Pandas / Parquet                     | Curated 95,090 film dataset from IMDb, TMDb & MovieLens            |

---

## 📖 Documentation & Deep Dives

* [Detailed Technical Architecture](./movie_recommendation_rag_system.md)
* [Hybrid Strategy Deep-Dive](<./🎬%20Hybrid%20Movie%20Recommendation%20Sy.md>)
* [Recommendation Strategy Catalog](./docs/recommendation_strategy_catalog.md)
