"""
Interactive test / demo for the semantic search + hybrid querying layer.

Prerequisite: run pipeline.py first so that vector_db.duckdb and
data_outputs/gold/gold_ai_articles_enriched.csv exist.

Run:
    .\\venv\\Scripts\\python.exe query_semantic_search.py
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from semantic_search import SemanticSearchEngine

ENRICHED_CSV = "./data_outputs/gold/gold_ai_articles_enriched.csv"
DB_PATH = "./vector_db.duckdb"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def demo_find_similar(engine):
    """Requirement: find_similar_articles(query_text, top_k=5) -> IDs + scores."""
    print("\n" + "=" * 80)
    print("DEMO 1: find_similar_articles(query_text, top_k=5)")
    print("=" * 80)
    query = "AI startup building large language models for enterprises"
    results = engine.find_similar_articles(query, top_k=5, threshold=0.0)
    print(f"\nQuery: {query!r}\n")
    for rank, (aid, score) in enumerate(results, 1):
        print(f"  {rank}. {aid}   similarity={score:.3f}")


def demo_similar_to_article(engine, df):
    """Find the articles most similar to an existing article (excluding itself)."""
    print("\n" + "=" * 80)
    print("DEMO 2: Articles similar to an existing article")
    print("=" * 80)
    row = df.iloc[0]
    aid = row["article_id"]
    query = f"{row['title']} {row['summary']}"
    print(f"\nReference: {aid} - {row['title']}\n")
    results = engine.find_similar_articles(query, top_k=6, threshold=0.0)
    results = [(rid, s) for rid, s in results if rid != aid][:5]
    for rank, (rid, score) in enumerate(results, 1):
        match = df.loc[df.article_id == rid, "title"]
        title = match.values[0] if len(match) else "(not in AI subset)"
        print(f"  {rank}. {rid}   similarity={score:.3f}   {title}")


def demo_hybrid(engine, df):
    """Requirement: hybrid search = SQL metadata filters + vector similarity.
    Example: AI articles from 2022-2024 with ARR > $50M, similar to a query."""
    print("\n" + "=" * 80)
    print("DEMO 3: Hybrid search (SQL filters + vector similarity)")
    print("Filter: published 2022-2024 AND arr_usd > $50M")
    print("=" * 80)

    # SQL filter stage (runs in DuckDB, reusing the engine's connection)
    engine.conn.register("ai_articles", df)
    filtered = engine.conn.execute(
        """
        SELECT article_id, title, arr_usd, published_date
        FROM ai_articles
        WHERE CAST(substr(CAST(published_date AS VARCHAR), 1, 4) AS INTEGER)
              BETWEEN 2022 AND 2024
          AND arr_usd > 50000000
        """
    ).fetchdf()
    engine.conn.unregister("ai_articles")
    print(f"\nSQL filter matched {len(filtered)} articles")

    if filtered.empty:
        print("No rows matched the filter.")
        return

    # Vector similarity stage (rank the filtered set by cosine similarity)
    query = "enterprise AI platform with strong revenue growth"
    qvec = next(iter(engine.model.embed([query])))
    cand_ids = [a for a in filtered["article_id"].tolist() if a in engine.embeddings_dict]
    cand_vecs = np.array([engine.embeddings_dict[a] for a in cand_ids])
    sims = cosine_similarity([qvec], cand_vecs)[0]
    ranked = sorted(zip(cand_ids, sims), key=lambda x: x[1], reverse=True)[:5]

    print(f"Query: {query!r}\n")
    print("Top 5 (filtered, then ranked by similarity):")
    for rank, (aid, score) in enumerate(ranked, 1):
        r = filtered.loc[filtered.article_id == aid].iloc[0]
        print(f"  {rank}. {aid}  sim={score:.3f}  ARR=${int(r.arr_usd):,}  {r.title}")


def main():
    df = pd.read_csv(ENRICHED_CSV)

    engine = SemanticSearchEngine(model_name=MODEL_NAME, db_path=DB_PATH)
    engine.load_embeddings_from_duckdb()

    demo_find_similar(engine)
    demo_similar_to_article(engine, df)
    demo_hybrid(engine, df)

    engine.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
