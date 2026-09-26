# Data Architecture

Design and table definitions for a PySpark pipeline.

---

## Design Philosophy

**Source lineage** — every modeled value links back to its raw source (`revenue_raw`, raw date/company fields) so parses can be audited and debugged.

**Layered processing (Bronze → Silver → Gold)**
- **Bronze**: raw data + parsing attempts only (no business logic)
- **Silver**: company matching, validation flags, metadata enrichment
- **Gold**: analytical star schema, marts, and data-quality audits

**Star schema** — a single fact table of ARR observations references a company dimension, avoiding attribute duplication and giving explicit FK lineage.

**Idempotent re-runs** — the fact grain is keyed by `article_id`; exports clean prior files before writing, so re-running produces no duplicates.

**Separation of concerns** — parsing lives in Bronze, matching/validation in Silver, sizing/analytics in Gold.

---

## Layer & Table Definitions

### BRONZE

#### bronze_articles (750)
Raw articles with parsing columns added; source columns preserved.

| Column | Type | Description |
|--------|------|-------------|
| article_id | string | Article identifier (PK) |
| title, company_name, summary, url | string | Source fields (raw) |
| published_date | string | Original date string |
| category | string | Original category string |
| revenue | string | Original revenue string (messy) |
| ingestion_date | timestamp | Ingestion time |
| arr_usd_parsed | long | Parsed ARR in USD (NULL if invalid) |
| published_date_parsed | date | Parsed date (NULL if invalid) |
| category_standardized | string | Normalized category |

#### bronze_companies (21)
Company metadata as-is, plus `ingestion_date`.

| Column | Type | Description |
|--------|------|-------------|
| company_name | string | Company (PK) |
| founded_year | integer | Year founded |
| headquarters | string | HQ location |
| employee_count | integer | Employees |
| industry | string | Industry |
| is_public | boolean | Public flag |
| stock_ticker | string | Ticker |
| ingestion_date | timestamp | Ingestion time |

---

### SILVER

#### silver_articles (750)
Articles after company matching, validation, and metadata enrichment. Company matching is resolved driver-side and attached via a native broadcast join (no UDFs — Windows-safe).

Key added columns:

| Column | Type | Description |
|--------|------|-------------|
| company_name_matched | string | Resolved company (FK → dim_company); NULL if unmatched |
| company_matched | boolean | Whether the company was resolved |
| arr_is_valid | boolean | Whether ARR parsed to a valid value |
| founded_year, headquarters, employee_count, industry, is_public, stock_ticker | — | Joined from metadata |
| company_age | integer | `article_year - founded_year` |

Plus all Bronze article columns (parsed values, lineage). Rows are retained even when unmatched or ARR-invalid, so Silver is the complete, flagged record set.

---

### GOLD (star schema + marts + audits)

#### fact_arr_observations (558) — FACT
One row per article with a valid ARR observation. Grain = article.

| Column | Type | Description |
|--------|------|-------------|
| article_id | string | Source article (PK / degenerate) |
| company_name | string | FK → dim_company (matched name) |
| observation_date | date | ARR as of the article's published_date |
| observation_year | integer | Year of observation (for filtering/analysis) |
| observation_quarter | integer | Quarter (1–4) of observation |
| observation_month | integer | Month (1–12) of observation |
| arr_usd | long | Normalized ARR in USD (measure) |
| company_age | integer | Company age at observation |
| category | string | Standardized category (degenerate) |
| revenue_raw | string | Original revenue string (lineage) |
| title, summary, url | string | Article context |

Only valid ARR rows are included (`arr_is_valid = true`); invalid/undisclosed values create no fact record.

#### dim_company (21) — DIMENSION
Company master, keyed by `company_name`.

| Column | Type | Description |
|--------|------|-------------|
| company_name | string | PK |
| founded_year, headquarters, employee_count, industry, is_public, stock_ticker | — | Static attributes |
| company_size_category | string | Small (<10k) / Medium (10k–30k) / Large (≥30k) |

#### gold_arr_latest (21) — DERIVED VIEW
Most recent ARR per company via `ROW_NUMBER() OVER (PARTITION BY company_name ORDER BY observation_date DESC)`. **Excludes unmatched (null-company) observations.**

| Column | Type | Description |
|--------|------|-------------|
| company_name | string | FK → dim_company |
| latest_arr_date | date | Date of latest observation |
| latest_arr_usd | long | Latest ARR value |
| article_id | string | Source article |

#### ai_articles_enriched (116) — AI MART
AI/ML articles, 2022–2024, ARR > $50M, enriched with company attributes, sizing, and semantic-search fields.

Filters: `(category = 'AI_ML' OR industry LIKE '%AI%' OR industry LIKE '%ML%') AND year BETWEEN 2022 AND 2024 AND arr_usd >= 50,000,000`.

Columns: `article_id, title, company_name, published_date, category, arr_usd, summary, url, industry, founded_year, headquarters, employee_count, is_public, stock_ticker, company_age, company_size_category, top_similar_articles, embedding`.

- `top_similar_articles` — top 3 most similar articles (format `ID|score;ID|score;ID|score`)
- `embedding` — 384-dim vector serialized as JSON

#### unmatched_companies (6) — AUDIT
Article company names not resolved to metadata.

| Column | Type | Description |
|--------|------|-------------|
| raw_company_name | string | Unresolved article company |
| article_count | integer | Articles with this name |
| potential_match | string | Closest metadata candidate |
| match_score | integer | Similarity (0–100), below threshold |

#### invalid_records (209) — AUDIT
Articles whose ARR or date failed to parse; raw values preserved for triage.

| Column | Type | Description |
|--------|------|-------------|
| article_id | string | Article |
| company_name | string | Raw company |
| revenue_raw | string | Original revenue string |
| arr_usd_parsed | long | Parsed ARR (NULL when failed) |
| arr_parse_failed | boolean | ARR parse failed |
| published_date_raw | string | Original date string |
| published_date_parsed | date | Parsed date (NULL when failed) |
| date_parse_failed | boolean | Date parse failed |

---

## Semantic Search Layer

Implemented in `semantic_search.py`; embeddings persisted to `vector_db.duckdb`.

- **Model**: `all-MiniLM-L6-v2` via **fastembed** (ONNX runtime; no PyTorch)
- **Input**: `title + summary`
- **Storage**: DuckDB table `article_embeddings (article_id, title, summary, embedding FLOAT8[])` + JSON in the AI CSV
- **APIs**:
  - `find_similar_articles(query_text, top_k=5)` → `[(article_id, score)]` (cosine similarity)
  - `compute_top_similar_articles_per_article(top_k=3)` → per-article neighbours (excludes self)
  - `hybrid_search(query_text, sql_filter, top_k)` → SQL metadata filter + vector ranking
  - `load_embeddings_from_duckdb()` → rehydrate cache for querying without re-embedding

---

## Data Quality & Lineage

- **ARR**: missing / `N/A` / `Not disclosed` / non-numeric → NULL; never defaulted. Excluded from the fact table; captured in `invalid_records`.
- **Dates**: unparseable/missing → NULL; captured in `invalid_records`.
- **Company matching**: below-threshold matches (< 80%) → `company_matched = false`, kept in Silver, excluded from `gold_arr_latest`, and listed in `unmatched_companies`.

**Ambiguous numeric dates**: the date parser tries `MM/dd/yyyy` (US) before `dd/MM/yyyy` (EU), so a value like `01/02/2023` resolves to **January 2, 2023** (US convention). This precedence is a documented assumption and easy to change in `parse_date_spark`.

**Deriving ARR views**: the fact table exposes `observation_year/quarter/month`, so *latest ARR* comes from `gold_arr_latest` (window on `observation_date`), and *quarterly ARR* is a simple aggregation, e.g. `GROUP BY company_name, observation_year, observation_quarter` taking `MAX(arr_usd)` (or the latest within each quarter).

**Revenue lineage**: `tech_news.csv (revenue)` → `bronze_articles (arr_usd_parsed + revenue)` → `fact_arr_observations (arr_usd + revenue_raw)`.

---

## Key Design Decisions

1. **Star schema over a denormalized fact** — avoids duplicating company attributes across observations and scales as the company list grows; the `company_name` FK gives explicit lineage.
2. **Fuzzy matching at 80% with multi-signal scoring** — `max(token_set_ratio, space-stripped ratio)` handles subsets ("Meta AI Research" → "Meta AI") and spacing ("Open AI" → "OpenAI"); an alias map covers true abbreviations only.
3. **Range midpoint with per-number suffixes** — `$X.XM - $Y.YM` is averaged (the regex tolerates a suffix between the first number and the dash), matching the dataset's actual range format.
4. **Company age at observation time** — `article_year - founded_year`, enabling age-cohort analysis.
5. **Sizing/analytics in Gold, not Silver** — `company_size_category` is computed in the analytical layer; Silver stays focused on cleaning/validation.
6. **Driver-side CSV writes** — avoids Windows `winutils.exe` dependencies; safe given the small data volume.
7. **fastembed/ONNX over sentence-transformers/PyTorch** — lighter, no VC++/CUDA needs, and sidesteps the NumPy 2.x binary incompatibility (pinned `numpy==1.26.4`, `onnxruntime==1.17.0`).

---

## Running Reliably Beyond a Local Batch

The pipeline is intentionally designed as if new article batches keep arriving. The following describes how the same data model scales from a local script to a reliable, scheduled production system.

### 1. Incremental / idempotent loading
- **Grain keyed by `article_id`** makes loads naturally idempotent. In production, replace full CSV overwrites with a **MERGE/UPSERT** on `article_id` into a transactional table format (Delta Lake, Apache Iceberg, or Hudi) so re-processing a batch never duplicates facts.
- Add an **ingestion watermark** (e.g. max `ingestion_date` or a source offset) so each run processes only new/changed articles rather than the entire history.
- Keep late-arriving and corrected records safe by making Silver an **append + merge** rather than a full rebuild.

### 2. Storage & partitioning
- Persist Bronze/Silver/Gold as **partitioned Parquet** (e.g. partition facts by `year(observation_date)`) instead of single CSVs. CSV remains a convenient export, not the system of record.
- Store `dim_company` as a slowly-changing dimension (SCD‑2) if company attributes change over time, so historical facts keep their point-in-time context.

### 3. Orchestration & scheduling
- Wrap the stages (ingest → bronze → silver → gold → semantic) as tasks in an orchestrator (**Airflow, Dagster, or Prefect**) with explicit dependencies, retries, and backfill support.
- Make each stage independently runnable and restartable; the current function boundaries (`create_bronze_*`, `create_silver_*`, `create_*` gold builders) map directly to tasks.

### 4. Scaling the compute
- The logic is **pure native Spark** (no Python UDFs), so it scales out unchanged from `local[*]` to a cluster (EMR, Dataproc, Databricks) — only the master URL and resource config change.
- Company matching is resolved on a small driver-side broadcast map, which stays cheap as article volume grows because it scales with the number of *distinct* company names, not rows.
- Re-enable whole-stage codegen and adaptive execution in a cluster environment (they are disabled here only for Windows-local stability).

### 5. Data quality as a first-class gate
- The `invalid_records` and `unmatched_companies` audits become **monitored quality metrics**: alert when the invalid-parse rate or unmatched rate crosses a threshold.
- Add schema/constraint checks (e.g. Great Expectations or `pytest`-style data tests) between layers to fail fast on upstream drift.
- The existing 67-test suite runs in CI on every change; extend with contract tests on the source schema.

### 6. The semantic layer at scale
- Swap the local DuckDB store for a managed vector database (e.g. **Qdrant, pgvector, or Milvus**) for concurrent, low-latency similarity queries.
- Embeddings are deterministic per `title + summary`, so only embed **new/changed** articles and upsert by `article_id`.
- Batch the `top_similar_articles` computation, or serve it on-demand from the vector index for larger corpora (the current all-pairs matrix is fine for hundreds–thousands of rows).

### 7. Configuration, secrets, and observability
- Externalize paths, rates, thresholds, and filters (already centralized in `config.py`) into environment/config management per deployment.
- Add structured logging and run-level metrics (rows in/out per layer, parse rates, match rates, durations) for observability and lineage tracking.

**Summary**: the data model (star schema + audits + `article_id` grain) is already production-shaped. Moving beyond local batch is primarily about swapping CSV for a transactional lake format, adding incremental MERGE loads, scheduling via an orchestrator, and promoting the audit tables to monitored quality gates — no changes to the modeling logic are required.

---

## Query Patterns Supported

**ARR over time for a company** (most important)
```python
import pandas as pd
fact = pd.read_csv("data_outputs/gold/fact_arr_observations.csv")
fact[fact.company_name == "Snowflake"].sort_values("observation_date")
```

**Source article for an ARR observation**
```python
fact[fact.article_id == "ART0004"][["article_id", "company_name", "arr_usd", "revenue_raw", "url"]]
```

**Filter by date / category / industry / ARR threshold**
```python
ai = pd.read_csv("data_outputs/gold/gold_ai_articles_enriched.csv")
ai[(ai.arr_usd > 100_000_000) & (ai.company_size_category == "Large")]
```

**Latest ARR per company**
```python
pd.read_csv("data_outputs/gold/gold_arr_latest.csv")
```

**Semantic + hybrid search** — see `query_semantic_search.py`.
