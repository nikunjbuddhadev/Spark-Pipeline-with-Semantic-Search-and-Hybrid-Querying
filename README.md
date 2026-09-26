# YipitData PySpark Data Pipeline

A local PySpark pipeline that turns messy technology-news articles into a queryable, star-schema data warehouse of company ARR observations — plus a semantic search layer over the AI articles.

## Overview

- **Input**: 750 technology-news articles (`tech_news.csv`) + company metadata (`company_metadata.json`, 21 companies)
- **Output**: Bronze/Silver/Gold CSV layers, a company ARR star schema, data-quality audit files, and an AI-articles dataset enriched with embeddings and semantic-similarity links
- **Focus**: Handling messy inputs, preserving source lineage, and producing queryable, auditable outputs

## Architecture

```
Ingestion  ->  Bronze (raw + parsed)  ->  Silver (matched + validated)  ->  Gold (star schema + marts)  ->  Semantic Search
```

Outputs are written under `data_outputs/` in layer subfolders:

```
data_outputs/
├── bronze/
│   ├── bronze_articles.csv        (750)  raw articles + parsed columns
│   └── bronze_companies.csv       (21)   raw company metadata
├── silver/
│   └── silver_articles.csv        (750)  matched, validated, enriched
└── gold/
    ├── fact_arr_observations.csv  (558)  FACT: one row per valid ARR observation
    ├── dim_company.csv            (21)   DIM: company master + size category
    ├── gold_arr_latest.csv        (21)   latest ARR per company (matched only)
    ├── ai_articles_enriched.csv   (116) AI mart + embedding + top_similar_articles
    ├── unmatched_companies.csv    (6)    audit: article companies not in metadata
    └── invalid_records.csv        (209)  audit: failed ARR/date parses
```

A DuckDB vector store (`vector_db.duckdb`) holds the article embeddings for similarity search.

## Quick Start

### System Requirements
- **OS**: Windows, macOS, or Linux (developed/tested on Windows 10/11)
- **Python**: 3.11 (a `venv/` is already provisioned in this workspace)
- **Java**: JDK 17 (Amazon Corretto recommended) — required for the PySpark JVM. The pipeline auto-detects `JAVA_HOME` from common install locations, or set it yourself.
- **Memory**: 2 GB minimum for the driver (4 GB+ recommended)
- **Disk**: ~250 MB for dependencies + ~90 MB for the cached embedding model
- **Network**: needed only on the first run, to download the `all-MiniLM-L6-v2` model (cached thereafter)

### Install
```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

Key dependencies: `pyspark`, `fuzzywuzzy` + `python-Levenshtein` (matching), `fastembed` + `onnxruntime` (embeddings), `duckdb` (vector store), `pytest` + `pytest-cov` (tests).

### Run the pipeline
```powershell
.\venv\Scripts\python.exe pipeline.py
```

The first run downloads the `all-MiniLM-L6-v2` embedding model (~90 MB) once and caches it.

### Try semantic search
```powershell
.\venv\Scripts\python.exe query_semantic_search.py
```

### Run the tests
```powershell
.\venv\Scripts\python.exe -m pytest tests/ --cov --cov-report=term-missing
```

## Regenerating All CSV Outputs

Running the pipeline regenerates **every required CSV** from the two source files, idempotently — each run cleans the previous file before writing, so there are no duplicates:

```powershell
.\venv\Scripts\python.exe pipeline.py
```

This (re)creates the full `data_outputs/` tree:

| Layer | File | Rows | Contents |
|-------|------|------|----------|
| bronze | `bronze/bronze_articles.csv` | 750 | raw articles + parsed columns |
| bronze | `bronze/bronze_companies.csv` | 21 | raw company metadata |
| silver | `silver/silver_articles.csv` | 750 | matched, validated, enriched |
| gold | `gold/fact_arr_observations.csv` | 558 | **fact**: valid ARR observations |
| gold | `gold/dim_company.csv` | 21 | **dimension**: company master |
| gold | `gold/gold_arr_latest.csv` | 21 | latest ARR per company |
| gold | `gold/ai_articles_enriched.csv` | 116 | AI mart + `embedding` + `top_similar_articles` |
| gold | `gold/unmatched_companies.csv` | 6 | audit: unresolved company names |
| gold | `gold/invalid_records.csv` | 209 | audit: failed ARR/date parses |

The embedding vector store `vector_db.duckdb` is also refreshed on each run. To force a completely clean rebuild, delete `data_outputs/` and `vector_db.duckdb` first, then re-run.

## Example Queries / Usage

All modeled tables are plain CSVs — query them with pandas, DuckDB, or Spark.

**ARR over time for a company** (the primary use case)
```python
import pandas as pd
fact = pd.read_csv("data_outputs/gold/fact_arr_observations.csv")
fact[fact.company_name == "Snowflake"].sort_values("observation_date")[
    ["observation_date", "arr_usd", "article_id"]
]
```

**Find the source article for an ARR observation**
```python
fact[fact.article_id == "ART0004"][
    ["article_id", "company_name", "arr_usd", "revenue_raw", "url"]
]
```

**Filter by date, category, industry, and ARR threshold**
```python
ai = pd.read_csv("data_outputs/gold/ai_articles_enriched.csv")
ai[(ai.arr_usd > 100_000_000) & (ai.company_size_category == "Large")]
```

**Quarterly ARR** (the fact table carries `observation_year/quarter/month`)
```python
fact = pd.read_csv("data_outputs/gold/fact_arr_observations.csv")
fact.groupby(["company_name", "observation_year", "observation_quarter"])["arr_usd"].max()
```

**Latest ARR per company**
```python
pd.read_csv("data_outputs/gold/gold_arr_latest.csv").sort_values(
    "latest_arr_usd", ascending=False
)
```

**Semantic similarity search**
```python
from semantic_search import SemanticSearchEngine
engine = SemanticSearchEngine(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    db_path="./vector_db.duckdb",
)
engine.load_embeddings_from_duckdb()
engine.find_similar_articles("enterprise AI platform with strong revenue growth", top_k=5)
```

**Hybrid search (SQL filter + vector similarity)** — see [`query_semantic_search.py`](query_semantic_search.py) for a runnable demo of all three search modes.

## Data Cleaning

### Revenue / ARR
- **Currencies**: USD, EUR (×1.1), GBP (×1.27), JPY (÷150)
- **Formats**: `5.2B`, `$5,200,000,000`, `1.480 billion`, `500M USD`, `$980.0M`
- **Ranges**: `$38475.0M - $42525.0M` (each side carrying its own suffix) → **midpoint**
- **Invalid**: missing, empty, `N/A`, `Not disclosed`, `unknown` → `NULL` (never silently treated as a valid ARR)
- **Type**: 64-bit `LongType` (handles values above the 32-bit limit, e.g. `$16.8B`)

### Dates
- Handles ISO (`2023-05-12`), US (`02/23/2023`), EU (`23-08-2023`), `03 Oct 2023`, `February 07, 2022`, and more, via the LEGACY time parser
- Unparseable/missing → `NULL`
- **Extracted parts**: `observation_year`, `observation_quarter`, `observation_month` are derived onto the fact table for filtering and analysis
- **Ambiguous numeric dates**: parsing tries `MM/dd/yyyy` (US) before `dd/MM/yyyy` (EU), so `01/02/2023` resolves to **Jan 2, 2023** (US convention)

### Categories
Mapped into a consistent taxonomy, e.g. `AI`, `ML`, `Artificial Intelligence`, `Machine Learning`, `AI & ML`, `AI/ML` → `AI_ML`; `Financial Technology` → `FinTech`; `Cloud Computing` → `Cloud`; unknown → `Other`.

### Company matching (Silver-layer business logic)
Resolves article company names to the metadata master via a 3-layer approach:
1. **Normalization** (lowercase, punctuation → space, collapse whitespace)
2. **Alias map** for true abbreviations (`aws` → Amazon Web Services, `azure`/`msft` → Microsoft)
3. **Multi-signal fuzzy match**: `max(token_set_ratio, space-stripped ratio)`, accepted at **80%**

Matching resolves 722/750 articles. Unmatched names are exported to `unmatched_companies.csv` for triage.

### Enrichment
- Company metadata joined onto matched articles (industry, founded_year, headquarters, employee_count, is_public, stock_ticker)
- `company_age = article_year - founded_year`
- `company_size_category`: Small (< 10,000), Medium (10,000–30,000), Large (> 30,000) — computed in the Gold layer

## Star Schema (Gold)

- **Fact** — `fact_arr_observations` (grain: one article with a valid ARR). Carries measures (`arr_usd`), the `company_name` FK, `observation_date`, degenerate article attributes, and `revenue_raw` lineage.
- **Dimension** — `dim_company` (PK `company_name`) with static attributes and `company_size_category`.
- **Derived views** — `gold_arr_latest` (latest ARR per company via a window function; matched companies only) and `ai_articles_enriched` (the AI mart).

### AI articles export — `ai_articles_enriched.csv`
All articles where:
- category is AI/ML **OR** company industry contains AI/ML, **AND**
- published between 2022 and 2024, **AND**
- valid ARR ≥ $50M USD

Columns: `article_id, title, company_name, published_date, category, arr_usd, summary, url, industry, founded_year, headquarters, employee_count, is_public, stock_ticker, company_age, company_size_category, top_similar_articles, embedding`.

## Semantic Search (Bonus)

Implemented in `semantic_search.py` using **fastembed** (ONNX runtime — no PyTorch) with the **`all-MiniLM-L6-v2`** model:
- Embeddings generated over `title + summary` and stored in DuckDB (`vector_db.duckdb`) plus serialized JSON in the AI CSV
- `find_similar_articles(query_text, top_k=5)` — cosine-similarity search returning article IDs + scores
- `hybrid_search(query_text, sql_filter, top_k)` — SQL metadata filters + vector similarity (e.g. AI articles 2022–2024, ARR > $50M, similar to a query)
- `top_similar_articles` — the top 3 most similar articles per article (format: `ART0085|0.611;ART0463|0.610;ART0493|0.595`)

## Data Quality & Audit

- `unmatched_companies.csv` — distinct article company names not resolved to metadata, with article counts and the closest candidate + score.
- `invalid_records.csv` — every article whose ARR **or** date failed to parse, with the raw source values and `arr_parse_failed` / `date_parse_failed` flags (for debugging failed parses).
- Idempotent re-runs: exports clean prior files before writing; the fact grain is keyed by `article_id`.

## Tests & Coverage

67 tests (`tests/`) run on a local SparkSession:
- Unit tests for revenue/date/category parsing, company matching, and semantic-search helpers
- Integration tests driving the full bronze → silver → gold chain on small in-memory DataFrames
- `modeling.py` line coverage is ~86%

```powershell
.\venv\Scripts\python.exe -m pytest tests/ --cov --cov-report=term-missing --cov-report=html
```

## Project Structure

```
POC/
├── pipeline.py                 # Orchestration: env setup, layer calls, exports
├── ingestion.py                # Load tech_news.csv and company_metadata.json
├── modeling.py                 # Parsing, matching, bronze/silver/gold builders, audits
├── semantic_search.py          # Embeddings, cosine search, DuckDB, hybrid search
├── query_semantic_search.py    # Demo/test script for the search layer
├── config.py                   # Currency rates, aliases, thresholds, filters
├── requirements.txt
├── pytest.ini / .coveragerc
├── tests/                      # Unit + integration tests
├── tech_news.csv               # Input: articles
├── company_metadata.json       # Input: company metadata
├── data_outputs/               # Bronze/Silver/Gold CSV exports
└── vector_db.duckdb            # Embedding vector store (regeneratable)
```

## Configuration

Edit `config.py`:
- `CURRENCY_RATES` — EUR/GBP/JPY conversion factors
- `CATEGORY_MAPPING` — source label → taxonomy
- `COMPANY_ALIASES` — abbreviation → canonical company
- `COMPANY_MATCH_THRESHOLD` — fuzzy acceptance (default 80)
- `COMPANY_SIZE_THRESHOLDS` — employee-count buckets
- `AI_ARTICLES_FILTERS` — e.g. `min_arr_usd`

## Notes on Windows
- CSV writing is done driver-side (Python `csv`) to avoid `winutils.exe` issues.
- Whole-stage codegen is disabled and the LEGACY date parser is enabled for local stability.
- The embedding stack uses `fastembed` + `onnxruntime==1.17.0` + `numpy==1.26.4` (avoids the NumPy 2.x binary incompatibility).

See `DATA_ARCHITECTURE.md` for the full table definitions and design decisions.
