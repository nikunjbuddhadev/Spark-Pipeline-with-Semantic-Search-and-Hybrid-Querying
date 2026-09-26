"""
Semantic Search Layer for YipitData Pipeline

Provides:
- Embedding generation for articles (title + summary)
- Cosine similarity search
- DuckDB integration for vector storage and hybrid querying
- Top-K similar articles per article
"""

import numpy as np
import pandas as pd
from fastembed import TextEmbedding
from sklearn.metrics.pairwise import cosine_similarity
import duckdb
from pathlib import Path
from typing import List, Tuple, Dict


class SemanticSearchEngine:
    """Generate embeddings, store in DuckDB, and perform similarity search."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", db_path: str = "./vector_db.duckdb"):
        """
        Initialize semantic search engine.
        
        Args:
            model_name: Embedding model to use (runs via ONNX through fastembed, no PyTorch)
            db_path: Path to DuckDB database file
        """
        print(f"[LOAD] Loading embedding model: {model_name}")
        self.model = TextEmbedding(model_name=model_name)
        self.db_path = db_path
        self.embeddings_dict = {}  # Cache: article_id -> embedding array
        self.conn = duckdb.connect(db_path)
        self._initialize_db()
    
    def _initialize_db(self):
        """Create DuckDB schema for embeddings and similarity search."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS article_embeddings (
                article_id VARCHAR PRIMARY KEY,
                title VARCHAR,
                summary VARCHAR,
                embedding FLOAT8[],
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS article_similarity (
                article_id VARCHAR PRIMARY KEY,
                top_similar_articles VARCHAR[],  -- JSON-serialized list of [article_id, score, rank]
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        print("[OK] DuckDB schema initialized")
    
    def generate_embeddings(self, articles_df) -> pd.DataFrame:
        """
        Generate embeddings for articles (title + summary).
        
        Args:
            articles_df: DataFrame with article_id, title, summary columns
            
        Returns:
            DataFrame with article_id and embedding columns
        """
        print("[SETUP] Generating embeddings for articles...")
        
        # Combine title + summary for richer context
        articles_df = articles_df.copy()
        articles_df['text_for_embedding'] = (
            articles_df['title'].fillna('') + ' ' + 
            articles_df['summary'].fillna('')
        ).str.strip()
        
        # Generate embeddings (fastembed returns a generator of numpy arrays)
        texts = articles_df['text_for_embedding'].tolist()
        embeddings = list(self.model.embed(texts))
        
        # Create result DataFrame
        embeddings_df = pd.DataFrame({
            'article_id': articles_df['article_id'],
            'embedding': [emb.tolist() for emb in embeddings]  # Convert numpy -> list for storage
        })
        
        # Cache embeddings in memory
        for idx, row in embeddings_df.iterrows():
            self.embeddings_dict[row['article_id']] = np.array(row['embedding'])
        
        print(f"[OK] Generated embeddings for {len(embeddings_df)} articles")
        return embeddings_df
    
    def store_embeddings_in_duckdb(self, articles_df, embeddings_df):
        """
        Store embeddings and article metadata in DuckDB.
        
        Args:
            articles_df: Original articles DataFrame
            embeddings_df: DataFrame with article_id and embedding columns
        """
        # Merge to get metadata
        merged = articles_df[['article_id', 'title', 'summary']].merge(
            embeddings_df, on='article_id', how='inner'
        )
        
        # Clear old embeddings
        self.conn.execute("DELETE FROM article_embeddings")
        
        # Register the DataFrame so DuckDB can read it, then insert
        self.conn.register("merged_data", merged)
        self.conn.execute("""
            INSERT INTO article_embeddings (article_id, title, summary, embedding)
            SELECT article_id, title, summary, embedding FROM merged_data
        """)
        self.conn.unregister("merged_data")
        
        print(f"[OK] Stored {len(merged)} embeddings in DuckDB")
    
    def load_embeddings_from_duckdb(self):
        """Load stored embeddings from DuckDB back into the in-memory cache.
        Enables similarity search in a fresh session without re-embedding."""
        rows = self.conn.execute(
            "SELECT article_id, embedding FROM article_embeddings"
        ).fetchall()
        self.embeddings_dict = {aid: np.array(emb, dtype=np.float32) for aid, emb in rows}
        print(f"[OK] Loaded {len(self.embeddings_dict)} embeddings from DuckDB")
        return len(self.embeddings_dict)
    
    def find_similar_articles(self, query_text: str, top_k: int = 5, threshold: float = 0.5) -> List[Tuple[str, float]]:
        """
        Find top-K similar articles using cosine similarity.
        
        Args:
            query_text: Query text (title + summary or custom query)
            top_k: Number of top results to return
            threshold: Minimum similarity score (0-1)
            
        Returns:
            List of (article_id, similarity_score) tuples
        """
        # Encode query (fastembed returns a generator; take the single vector)
        query_embedding = next(iter(self.model.embed([query_text])))
        
        if not self.embeddings_dict:
            print("[WARN] No embeddings loaded. Generate embeddings first.")
            return []
        
        # Compute similarities
        article_ids = list(self.embeddings_dict.keys())
        embeddings = np.array([self.embeddings_dict[aid] for aid in article_ids])
        
        # Cosine similarity: (1 + score) / 2 to convert from [-1, 1] to [0, 1]
        similarities = cosine_similarity([query_embedding], embeddings)[0]
        
        # Sort by similarity
        results = sorted(
            zip(article_ids, similarities),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Filter by threshold and return top-k
        results = [(aid, score) for aid, score in results if score >= threshold][:top_k]
        return results
    
    def compute_top_similar_articles_per_article(self, top_k: int = 3) -> Dict[str, List[Tuple[str, float]]]:
        """
        For each article, find top-K most similar articles (excluding itself).
        
        Args:
            top_k: Number of similar articles to return per article
            
        Returns:
            Dict: {article_id: [(similar_article_id, score), ...]}
        """
        print(f"[SEARCH] Computing top-{top_k} similar articles for each article...")
        
        article_ids = list(self.embeddings_dict.keys())
        embeddings = np.array([self.embeddings_dict[aid] for aid in article_ids])
        
        # Compute pairwise cosine similarity matrix
        similarity_matrix = cosine_similarity(embeddings, embeddings)
        
        top_similar_dict = {}
        for i, article_id in enumerate(article_ids):
            # Get similarities for this article
            sims = similarity_matrix[i]
            
            # Sort by similarity (excluding self, which has similarity ~1.0)
            similar_pairs = [
                (article_ids[j], float(sims[j]))
                for j in range(len(article_ids))
                if j != i  # Exclude self
            ]
            similar_pairs.sort(key=lambda x: x[1], reverse=True)
            
            # Take top-k
            top_similar_dict[article_id] = similar_pairs[:top_k]
        
        print(f"[OK] Computed top-{top_k} similar articles for {len(top_similar_dict)} articles")
        return top_similar_dict
    
    def add_top_similar_articles_to_dataframe(self, df: pd.DataFrame, top_similar_dict: Dict) -> pd.DataFrame:
        """
        Add top_similar_articles column to DataFrame.
        
        Args:
            df: DataFrame with article_id column
            top_similar_dict: Dict from compute_top_similar_articles_per_article()
            
        Returns:
            DataFrame with new top_similar_articles column
        """
        df = df.copy()
        
        # Create column with list of similar article IDs and scores
        def get_similar_articles(article_id):
            if article_id in top_similar_dict:
                similar = top_similar_dict[article_id]
                # Return as string: "article_id1|score1;article_id2|score2;..."
                return ";".join([f"{aid}|{score:.3f}" for aid, score in similar])
            return ""
        
        df['top_similar_articles'] = df['article_id'].apply(get_similar_articles)
        return df
    
    def hybrid_search(self, 
                     query_text: str = None,
                     sql_filter: str = None,
                     top_k: int = 10) -> List[Dict]:
        """
        Hybrid search: SQL filters + vector similarity.
        
        Example:
            hybrid_search(
                query_text="AI companies with high ARR",
                sql_filter="WHERE category = 'AI_ML' AND arr_usd > 50000000",
                top_k=5
            )
        
        Args:
            query_text: Text query for semantic similarity
            sql_filter: SQL WHERE clause for metadata filtering
            top_k: Number of results to return
            
        Returns:
            List of article records (dict) sorted by similarity
        """
        print(f"[SEARCH] Performing hybrid search: query='{query_text}', filter='{sql_filter}'")
        
        if not query_text:
            print("[WARN] Hybrid search requires query_text. Falling back to SQL-only.")
            if sql_filter:
                results = self.conn.execute(f"SELECT * FROM article_embeddings {sql_filter} LIMIT {top_k}").fetchall()
                return results
            return []
        
        # Find similar articles
        similar_articles = self.find_similar_articles(query_text, top_k=1000)  # Get many candidates
        similar_ids = [aid for aid, _ in similar_articles]
        
        # Build SQL query with similarity filter
        if similar_ids:
            ids_str = "', '".join(similar_ids)
            sql = f"""
                SELECT article_id, title, summary, embedding
                FROM article_embeddings
                WHERE article_id IN ('{ids_str}')
            """
            if sql_filter:
                sql += f" {sql_filter}"
            
            sql += f" LIMIT {top_k}"
            results = self.conn.execute(sql).fetchall()
        else:
            results = []
        
        print(f"[OK] Hybrid search returned {len(results)} results")
        return results
    
    def close(self):
        """Close DuckDB connection."""
        if self.conn:
            self.conn.close()
            print("[OK] DuckDB connection closed")


def create_embedding_column(df: pd.DataFrame, embeddings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge embeddings into DataFrame for CSV export.
    
    Args:
        df: Main DataFrame
        embeddings_df: DataFrame with article_id and embedding columns
        
    Returns:
        DataFrame with embedding column (as JSON string for CSV export)
    """
    import json
    df = df.copy()
    
    # Merge embeddings
    merged = df.merge(embeddings_df, on='article_id', how='left')
    
    # Convert embedding arrays to JSON strings for CSV export
    merged['embedding'] = merged['embedding'].apply(
        lambda x: json.dumps(x.tolist()) if isinstance(x, np.ndarray) else (json.dumps(x) if isinstance(x, list) else None)
    )
    
    return merged
