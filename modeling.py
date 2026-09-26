from pyspark.sql import functions as F, Window
from pyspark.sql.types import IntegerType, LongType, StringType, BooleanType, DoubleType, StructType, StructField
from config import COMPANY_SIZE_THRESHOLDS, AI_ARTICLES_FILTERS, COMPANY_ALIASES, COMPANY_MATCH_THRESHOLD
from datetime import datetime
import re
from fuzzywuzzy import fuzz

def parse_revenue_spark(revenue_col):
    """
    Parse revenue using native Spark SQL functions (scalable, no UDF)
    Returns Column expression for arr_usd (integer USD amount)
    """
    
    # Handle null/empty/invalid
    cleaned_revenue = F.trim(revenue_col)
    is_invalid = (
        cleaned_revenue.isNull() | 
        (cleaned_revenue == "") |
        cleaned_revenue.rlike("(?i)(not disclosed|unknown|undisclosed|n/a)")
    )
    
    # Remove currency symbols and convert to lowercase for processing
    stripped = F.lower(
        F.regexp_replace(
            F.regexp_replace(
                F.regexp_replace(
                    F.regexp_replace(cleaned_revenue, "[£€¥$]", ""),
                    "usd", ""
                ),
                ",", ""
            ),
            r"\s+", " "
        )
    )
    
    # Detect currency
    has_gbp = revenue_col.contains("£") | revenue_col.contains("GBP")
    has_eur = revenue_col.contains("€") | revenue_col.contains("EUR")
    has_jpy = revenue_col.contains("¥") | revenue_col.contains("JPY")
    
    # Extract numeric value (first number found)
    numeric_str = F.regexp_extract(stripped, r"(\d+\.?\d*)", 1)
    numeric_val = F.when(numeric_str != "", numeric_str.cast(DoubleType())).otherwise(None)
    
    # Detect multiplier suffix
    has_billion = stripped.rlike("(?i)(b|billion)")
    has_million = stripped.rlike("(?i)(m|million)")
    has_thousand = stripped.rlike("(?i)(k|thousand)")
    
    multiplier = F.when(has_billion, 1_000_000_000.0) \
        .when(has_million, 1_000_000.0) \
        .when(has_thousand, 1_000.0) \
        .otherwise(1.0)
    
    # Handle ranges like "10M - 20M" or "38475.0M - 42525.0M" (each number may
    # carry its own multiplier suffix). Allow optional suffix letters between the
    # first number and the dash so the range is detected and averaged (midpoint).
    _range_re = r"(\d+\.?\d*)\s*[a-z]*\s*[-–]\s*(\d+\.?\d*)"
    range_lower = F.regexp_extract(stripped, _range_re, 1)
    range_upper = F.regexp_extract(stripped, _range_re, 2)
    has_range = stripped.rlike(_range_re)
    
    # Calculate amount
    amount = F.when(
        has_range & (range_lower != "") & (range_upper != ""),
        ((range_lower.cast(DoubleType()) + range_upper.cast(DoubleType())) / 2.0) * multiplier
    ).when(
        numeric_val.isNotNull() & ~has_range,
        numeric_val * multiplier
    ).otherwise(None)
    
    # Apply currency conversion
    usd_amount = F.when(
        has_gbp,
        amount * 1.27
    ).when(
        has_eur,
        amount * 1.1
    ).when(
        has_jpy,
        amount / 150.0
    ).otherwise(amount)
    
    # Final result: cast to long (64-bit) to hold billions, or null if invalid
    result = F.when(
        ~is_invalid & (usd_amount.isNotNull()) & (usd_amount > 0),
        usd_amount.cast(LongType())
    ).otherwise(None)
    
    return result


def parse_date_spark(date_col):
    """
    Parse date using native Spark SQL functions (scalable, no UDF)
    Tries multiple date formats. Returns date or null.
    """
    
    cleaned = F.trim(date_col)
    
    # Try multiple date formats with coalesce
    parsed = F.coalesce(
        F.to_date(cleaned, "yyyy-MM-dd"),           # ISO
        F.to_date(cleaned, "MM/dd/yyyy"),           # US
        F.to_date(cleaned, "dd/MM/yyyy"),           # EU
        F.to_date(cleaned, "dd MMM yyyy"),          # 21 Feb 2020
        F.to_date(cleaned, "dd-MM-yyyy"),           # 21-02-2020
        F.to_date(cleaned, "MMM dd, yyyy"),         # Feb 21, 2020
        F.to_date(cleaned, "MMMM dd, yyyy"),        # February 21, 2020
        F.to_date(cleaned, "yyyy/MM/dd"),           # 2020/02/21
    )
    
    return parsed


def standardize_category_spark(category_col):
    """
    Standardize category using native Spark SQL (scalable, no UDF)
    Maps to consistent taxonomy
    """
    
    lower_cat = F.lower(F.trim(category_col))
    
    standardized = F.when(
        lower_cat.rlike("(ai|ml|machine learning|artificial intelligence|ai & ml|ai/ml)"),
        "AI_ML"
    ).when(
        lower_cat.rlike("(fintech|financial technology)"),
        "FinTech"
    ).when(
        lower_cat.rlike("(cloud computing)"),
        "Cloud"
    ).when(
        lower_cat.rlike("(security)"),
        "Security"
    ).when(
        lower_cat.rlike("(data analytics|analytics)"),
        "Data Analytics"
    ).when(
        lower_cat.rlike("(software)"),
        "Software"
    ).when(
        lower_cat.rlike("(saas|software as a service)"),
        "SaaS"
    ).otherwise("Other")
    
    return standardized


def create_bronze_articles(articles_df):
    """
    Create BRONZE layer: Raw articles with minimal cleanup
    Preserve original fields + add parsing attempt columns
    Uses native Spark SQL functions (scalable, no UDFs except fuzzy matching)
    """
    
    print("\n[SETUP] Creating BRONZE_ARTICLES layer...")
    
    # Add ingestion timestamp
    bronze_df = articles_df.withColumn("ingestion_date", F.current_timestamp())
    
    # Parse revenue using native Spark functions
    bronze_df = bronze_df.withColumn(
        "arr_usd_parsed",
        parse_revenue_spark(F.col("revenue"))
    )
    
    # Parse date using native Spark functions
    bronze_df = bronze_df.withColumn(
        "published_date_parsed",
        parse_date_spark(F.col("published_date"))
    )
    
    # Standardize category using native Spark functions
    bronze_df = bronze_df.withColumn(
        "category_standardized",
        standardize_category_spark(F.col("category"))
    )
    
    print(f"[OK] Bronze articles: {bronze_df.count()} rows")
    return bronze_df


def create_bronze_companies(companies_df):
    """Create BRONZE layer: Companies with minimal transformation"""
    print("\n[SETUP] Creating BRONZE_COMPANIES layer...")
    bronze_df = companies_df.withColumn("ingestion_date", F.current_timestamp())
    print(f"[OK] Bronze companies: {bronze_df.count()} rows")
    return bronze_df


def normalize_company_name(name):
    """Lowercase, strip punctuation to spaces, and collapse whitespace."""
    s = (name or "").lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _company_similarity(norm_a, norm_c):
    """
    Best of two signals:
    - token_set_ratio on normalized names -> handles subset/suffix (Meta AI Research -> Meta AI)
    - ratio on space-stripped names -> handles spacing variants (Open AI -> OpenAI)
    """
    return max(
        fuzz.token_set_ratio(norm_a, norm_c),
        fuzz.ratio(norm_a.replace(" ", ""), norm_c.replace(" ", "")),
    )


def best_company_candidate(article_name, company_names_list):
    """Return (best_company, score) for the closest metadata company, ignoring the threshold."""
    norm_a = normalize_company_name(article_name)
    if not norm_a:
        return None, 0
    best, best_score = None, 0
    for company in company_names_list:
        score = _company_similarity(norm_a, normalize_company_name(company))
        if score > best_score:
            best_score = score
            best = company
    return best, best_score


def match_company_names(article_name, company_names_list):
    """
    Resolve an article company name to a known company (driver-side) via:
    1) normalization, 2) exact match, 3) acronym/alias map,
    4) token_set_ratio + space-stripped ratio, accepted at COMPANY_MATCH_THRESHOLD.
    """
    norm_a = normalize_company_name(article_name)
    if not norm_a:
        return None

    if norm_a in COMPANY_ALIASES:
        return COMPANY_ALIASES[norm_a]

    best_match = None
    best_score = 0
    for company in company_names_list:
        norm_c = normalize_company_name(company)
        if norm_a == norm_c:
            return company
        score = _company_similarity(norm_a, norm_c)
        if score > best_score:
            best_score = score
            best_match = company

    return best_match if best_score >= COMPANY_MATCH_THRESHOLD else None


def build_company_match_map(bronze_articles_df, bronze_companies_df, spark):
    """
    Resolve each distinct article company name to a known company on the driver,
    then return a small Spark DataFrame [company_name_raw, company_name_matched]
    for a native broadcast join. Avoids Python-worker UDFs (fragile on Windows).
    """
    known_companies = [row[0] for row in bronze_companies_df.select("company_name").collect()]
    distinct_raw = [row[0] for row in bronze_articles_df.select("company_name").distinct().collect()]

    mapping = [(raw, match_company_names(raw, known_companies)) for raw in distinct_raw]

    return spark.createDataFrame(mapping, schema=StructType([
        StructField("company_name_raw", StringType(), True),
        StructField("company_name_matched", StringType(), True),
    ]))


def create_unmatched_companies_report(bronze_articles_df, bronze_companies_df, spark):
    """
    UNMATCHED_COMPANIES: distinct article company names that did not resolve to a
    metadata company, with article counts and the closest candidate for triage.
    """
    print("\n[SETUP] Creating UNMATCHED_COMPANIES report...")

    known = [row[0] for row in bronze_companies_df.select("company_name").collect()]
    counts = bronze_articles_df.groupBy("company_name").count().collect()

    rows = []
    for r in counts:
        raw = r["company_name"]
        if match_company_names(raw, known) is not None:
            continue  # resolved -> not part of the unmatched report
        candidate, score = best_company_candidate(raw, known)
        rows.append((raw, int(r["count"]), candidate, int(score)))

    rows.sort(key=lambda x: x[1], reverse=True)

    report_df = spark.createDataFrame(rows, schema=StructType([
        StructField("raw_company_name", StringType(), True),
        StructField("article_count", IntegerType(), True),
        StructField("potential_match", StringType(), True),
        StructField("match_score", IntegerType(), True),
    ]))

    print(f"[OK] Unmatched companies: {report_df.count()} distinct names")
    return report_df


def create_invalid_records_report(bronze_articles_df):
    """
    INVALID_RECORDS: articles whose revenue could not be parsed into a valid ARR
    or whose published_date could not be parsed. Preserves the raw source values
    so failed parses can be triaged. Grain = article.
    """
    print("\n[SETUP] Creating INVALID_RECORDS report...")

    invalid_df = bronze_articles_df.filter(
        F.col("arr_usd_parsed").isNull() | F.col("published_date_parsed").isNull()
    ).select(
        F.col("article_id"),
        F.col("company_name"),
        F.col("revenue").alias("revenue_raw"),
        F.col("arr_usd_parsed"),
        F.col("arr_usd_parsed").isNull().alias("arr_parse_failed"),
        F.col("published_date").alias("published_date_raw"),
        F.col("published_date_parsed"),
        F.col("published_date_parsed").isNull().alias("date_parse_failed"),
    )

    print(f"[OK] Invalid records: {invalid_df.count()} rows")
    return invalid_df


def create_silver_articles(bronze_articles_df, bronze_companies_df, spark):
    """
    Create SILVER layer: cleaned, validated, enriched articles.
    Company matching is resolved driver-side and attached via a native broadcast join
    (no UDFs), so the whole layer runs on native Spark.
    """

    print("\n[SETUP] Creating SILVER_ARTICLES layer...")

    # Resolve raw -> matched company names once, then broadcast join
    match_map_df = build_company_match_map(bronze_articles_df, bronze_companies_df, spark)

    silver_df = bronze_articles_df.join(
        F.broadcast(match_map_df),
        bronze_articles_df["company_name"] == match_map_df["company_name_raw"],
        "left"
    ).drop("company_name_raw")

    # Add validation flag
    silver_df = silver_df.withColumn(
        "company_matched",
        F.col("company_name_matched").isNotNull()
    )
    
    # Flag valid ARR (must be non-null after parsing)
    silver_df = silver_df.withColumn(
        "arr_is_valid",
        F.col("arr_usd_parsed").isNotNull()
    )
    
    # Join with company metadata
    silver_df = silver_df.join(
        bronze_companies_df,
        silver_df["company_name_matched"] == bronze_companies_df["company_name"],
        "left"
    )
    
    # Add company_age (as of article published_date)
    silver_df = silver_df.withColumn(
        "article_year",
        F.year(F.col("published_date_parsed"))
    ).withColumn(
        "company_age",
        F.when(F.col("founded_year").isNotNull(), F.col("article_year") - F.col("founded_year"))
        .otherwise(None)
    ).drop("article_year")
    
    print(f"[OK] Silver articles: {silver_df.count()} rows")
    print(f"  - Valid ARR observations: {silver_df.filter(F.col('arr_is_valid')).count()}")
    print(f"  - Matched companies: {silver_df.filter(F.col('company_matched')).count()}")
    
    return silver_df


def create_fact_arr_observations(silver_articles_df):
    """
    FACT_ARR_OBSERVATIONS: one row per article with a valid ARR observation.
    Grain = article. Carries measures, the company_name FK, observation-time
    attributes, degenerate article attributes, and source lineage.
    Static company attributes live in DIM_COMPANY.
    """

    print("\n[SETUP] Creating FACT_ARR_OBSERVATIONS...")

    arr_obs_df = silver_articles_df.filter(F.col("arr_is_valid")).select(
        F.col("article_id"),
        F.col("company_name_matched").alias("company_name"),  # FK -> dim_company
        F.col("published_date_parsed").alias("observation_date"),
        F.year(F.col("published_date_parsed")).alias("observation_year"),
        F.quarter(F.col("published_date_parsed")).alias("observation_quarter"),
        F.month(F.col("published_date_parsed")).alias("observation_month"),
        F.col("arr_usd_parsed").alias("arr_usd"),
        F.col("company_age"),  # observation-time attribute (as of published_date)
        F.col("category_standardized").alias("category"),
        F.col("revenue").alias("revenue_raw"),  # source lineage
        F.col("title"),
        F.col("summary"),
        F.col("url")
    )

    print(f"[OK] Fact ARR observations: {arr_obs_df.count()} rows")
    return arr_obs_df


def create_gold_arr_latest(fact_arr_obs_df):
    """
    GOLD_ARR_LATEST: most recent ARR observation per company (derived view).
    Attributes come from DIM_COMPANY via the company_name FK.
    """

    print("\n[SETUP] Creating GOLD_ARR_LATEST...")

    # Only known companies belong in the latest-ARR view; drop unmatched (null FK).
    matched_obs = fact_arr_obs_df.filter(F.col("company_name").isNotNull())

    window_spec = Window.partitionBy("company_name").orderBy(F.desc("observation_date"))

    gold_latest = matched_obs \
        .withColumn("rn", F.row_number().over(window_spec)) \
        .filter(F.col("rn") == 1) \
        .select(
            F.col("company_name"),
            F.col("observation_date").alias("latest_arr_date"),
            F.col("arr_usd").alias("latest_arr_usd"),
            F.col("article_id")  # source article for this ARR
        )

    print(f"[OK] Gold latest ARR: {gold_latest.count()} rows")
    return gold_latest


def create_gold_ai_articles_enriched(silver_articles_df):
    """
    Create GOLD_AI_ARTICLES_ENRICHED: Filtered AI articles (2022-2024, ARR > $50M)
    """
    
    print("\n[SETUP] Creating GOLD_AI_ARTICLES_ENRICHED layer...")
    
    # published_date_parsed is already a date; derive year for filtering
    ai_articles_df = silver_articles_df.withColumn(
        "year",
        F.year(F.col("published_date_parsed"))
    )
    
    # Apply filters
    filters_config = AI_ARTICLES_FILTERS
    
    ai_articles_df = ai_articles_df.filter(
        (
            (F.col("category_standardized") == "AI_ML") | 
            (F.col("industry").like("%AI%")) |
            (F.col("industry").like("%ML%"))
        ) &
        (F.col("year").between(2022, 2024)) &
        (F.col("arr_usd_parsed") >= filters_config['min_arr_usd'])
    )
    
    # Compute company_size_category from employee_count (not in silver layer).
    # Spec: Small < 10k, Medium 10k-30k inclusive, Large > 30k.
    ai_articles_df = ai_articles_df.withColumn(
        "company_size_category",
        F.when(F.col("employee_count") < 10000, "Small")
         .when(F.col("employee_count") <= 30000, "Medium")
         .when(F.col("employee_count") > 30000, "Large")
         .otherwise("Unknown")
    )
    
    # Select and rename columns as per requirements
    ai_articles_df = ai_articles_df.select(
        F.col("article_id"),
        F.col("title"),
        F.col("company_name_matched").alias("company_name"),
        F.col("published_date_parsed").alias("published_date"),
        F.col("category_standardized").alias("category"),
        F.col("arr_usd_parsed").alias("arr_usd"),
        F.col("summary"),
        F.col("url"),
        F.col("industry"),
        F.col("founded_year"),
        F.col("headquarters"),
        F.col("employee_count"),
        F.col("is_public"),
        F.col("stock_ticker"),
        F.col("company_age"),
        F.col("company_size_category")
    )
    
    print(f"[OK] Gold AI articles enriched: {ai_articles_df.count()} rows")
    return ai_articles_df


def create_gold_dim_company(bronze_companies_df):
    """
    DIM_COMPANY: company master dimension from the metadata, keyed by company_name.
    Adds company_size_category (static, from employee_count).
    """

    print("\n[SETUP] Creating DIM_COMPANY...")

    dim_company = bronze_companies_df.select(
        F.col("company_name"),
        F.col("founded_year"),
        F.col("headquarters"),
        F.col("employee_count"),
        F.col("industry"),
        F.col("is_public"),
        F.col("stock_ticker")
    ).withColumn(
        "company_size_category",
        F.when(F.col("employee_count") < 10000, "Small")
         .when(F.col("employee_count") <= 30000, "Medium")
         .when(F.col("employee_count") > 30000, "Large")
         .otherwise("Unknown")
    ).drop_duplicates()

    print(f"[OK] Dimension company: {dim_company.count()} rows")
    return dim_company


def enrich_with_semantic_search(gold_ai_articles_df):
    """
    Add embeddings and top_similar_articles to AI articles dataset.
    
    This function:
    1. Converts Spark DataFrame to Pandas
    2. Generates embeddings using sentence-transformers
    3. Computes top-3 similar articles per article
    4. Returns enriched Pandas DataFrame
    
    Args:
        gold_ai_articles_df: Spark DataFrame of AI articles
        
    Returns:
        Pandas DataFrame with embedding and top_similar_articles columns
    """
    print("\n[SEMANTIC] SEMANTIC SEARCH LAYER")
    print("=" * 80)
    
    from semantic_search import SemanticSearchEngine
    import pandas as pd
    
    # Convert to Pandas for embedding generation
    print("[LOAD] Converting AI articles to Pandas for embedding...")
    ai_articles_pd = gold_ai_articles_df.toPandas()
    
    if len(ai_articles_pd) == 0:
        print("[WARN] No AI articles to embed. Skipping semantic search.")
        return ai_articles_pd
    
    # Initialize semantic search engine
    search_engine = SemanticSearchEngine(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        db_path="./vector_db.duckdb"
    )
    
    # Generate embeddings
    embeddings_df = search_engine.generate_embeddings(ai_articles_pd)
    
    # Store embeddings in DuckDB
    search_engine.store_embeddings_in_duckdb(ai_articles_pd, embeddings_df)
    
    # Compute top-3 similar articles per article
    top_similar_dict = search_engine.compute_top_similar_articles_per_article(top_k=3)
    
    # Add to DataFrame
    ai_articles_enriched = search_engine.add_top_similar_articles_to_dataframe(
        ai_articles_pd, 
        top_similar_dict
    )
    
    # Merge embeddings column
    ai_articles_enriched = ai_articles_enriched.merge(
        embeddings_df, 
        on='article_id', 
        how='left'
    )
    
    # Convert embedding arrays to JSON strings for CSV export
    import json
    ai_articles_enriched['embedding'] = ai_articles_enriched['embedding'].apply(
        lambda x: json.dumps(x) if isinstance(x, list) else None
    )
    
    search_engine.close()
    
    print(f"[OK] Enriched {len(ai_articles_enriched)} AI articles with embeddings and similarity links")
    print("=" * 80)
    
    return ai_articles_enriched
