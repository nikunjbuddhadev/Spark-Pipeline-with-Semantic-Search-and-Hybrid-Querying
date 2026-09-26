import os
import sys
from pathlib import Path

# Force UTF-8 output so console prints (checkmarks, etc.) don't crash on Windows cp1252
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass

# Ensure Spark uses this interpreter for driver and workers (avoids Windows "python not found")
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

# Point Spark at winutils.exe so it can write files on Windows (Hadoop FileSystem needs it)
_HADOOP_HOME = r"C:\hadoop"
if os.path.isdir(_HADOOP_HOME):
    os.environ.setdefault("HADOOP_HOME", _HADOOP_HOME)
    os.environ["PATH"] = os.path.join(_HADOOP_HOME, "bin") + os.pathsep + os.environ.get("PATH", "")

from pyspark.sql import SparkSession
from ingestion import read_tech_news_csv, read_company_metadata_json
from modeling import (
    create_bronze_articles,
    create_bronze_companies,
    create_silver_articles,
    create_fact_arr_observations,
    create_gold_arr_latest,
    create_gold_ai_articles_enriched,
    create_gold_dim_company,
    create_unmatched_companies_report,
    create_invalid_records_report,
    enrich_with_semantic_search,
)


def setup_java_home():
    """Ensure JAVA_HOME is set for Spark Java gateway to work on Windows"""
    if "JAVA_HOME" in os.environ:
        return  # Already set, nothing to do
    
    # Search for Java installations in common Windows locations
    common_java_paths = [
        r"C:\Program Files\Amazon Corretto",
        r"C:\Program Files\Java",
        r"C:\Program Files (x86)\Java",
    ]
    
    for base_path in common_java_paths:
        if os.path.isdir(base_path):
            for jdk_dir in os.listdir(base_path):
                jdk_path = os.path.join(base_path, jdk_dir)
                if os.path.isdir(jdk_path) and os.path.isdir(os.path.join(jdk_path, "bin")):
                    os.environ["JAVA_HOME"] = jdk_path
                    return
    
    # If no Java found, Spark will fail with a clear error message


def initialize_spark():
    """Initialize Spark Session for local execution"""
    spark = SparkSession.builder \
        .appName("Spark_Pipeline") \
        .master("local[2]") \
        .config("spark.sql.shuffle.partitions", "4") \
        .config("spark.driver.memory", "2g") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
        .config("spark.sql.codegen.wholeStage", "false") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    print("[OK] Spark Session initialized")
    return spark


def ensure_output_directories(base_output_dir="./data_outputs"):
    """Create output directories and layer subdirectories if they don't exist"""
    Path(base_output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create layer subdirectories
    layers = ["bronze", "silver", "gold"]
    layer_dirs = {}
    for layer in layers:
        layer_path = os.path.join(base_output_dir, layer)
        Path(layer_path).mkdir(parents=True, exist_ok=True)
        layer_dirs[layer] = layer_path
    
    print(f"[OK] Output directory structure created: {base_output_dir}/")
    for layer in layers:
        print(f"  |- {layer}/")
    return base_output_dir, layer_dirs


def write_dataframe_to_csv(df, output_path):
    """
    Write a Spark DataFrame to a single CSV file from the driver.
    Avoids Spark's Hadoop writer (winutils.exe), which is unreliable on Windows.
    Data volumes here are small, so collecting to the driver is safe.
    """
    import csv
    import shutil
    # Clean up any leftover file/dir at the target path (idempotent re-runs)
    if os.path.isdir(output_path):
        shutil.rmtree(output_path, ignore_errors=True)
    elif os.path.exists(output_path):
        os.remove(output_path)

    columns = df.columns
    rows = df.collect()
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([row[c] for c in columns])
    return len(rows)


def write_pandas_dataframe_to_csv(df, output_path):
    """
    Write a Pandas DataFrame to CSV file.
    Used for semantic search enriched data which is already in Pandas format.
    """
    import shutil
    # Clean up any leftover file at the target path (idempotent re-runs)
    if os.path.isdir(output_path):
        shutil.rmtree(output_path, ignore_errors=True)
    elif os.path.exists(output_path):
        os.remove(output_path)
    
    df.to_csv(output_path, index=False, encoding='utf-8')
    return len(df)


def run_pipeline(tech_news_path, company_metadata_path, output_dir="./data_outputs"):
    """
    Run the complete pipeline
    
    Args:
        tech_news_path: Path to tech_news.csv
        company_metadata_path: Path to company_metadata.json
        output_dir: Directory to save output CSVs
    """
    
    print("\n" + "="*80)
    print(">>> PySpark Pipeline Starting...")
    print("="*80)
    
    # Setup Java environment for Spark gateway
    setup_java_home()
    
    # Initialize Spark
    spark = initialize_spark()
    output_dir, layer_dirs = ensure_output_directories(output_dir)
    
    try:
        # ============ INGESTION ============
        print("\n" + "="*80)
        print("[INGESTION LAYER]")
        print("="*80)
        articles_df = read_tech_news_csv(spark, tech_news_path)
        companies_df = read_company_metadata_json(spark, company_metadata_path)
        
        # ============ BRONZE LAYER ============
        print("\n" + "="*80)
        print("[BRONZE LAYER - Raw Data]")
        print("="*80)
        bronze_articles = create_bronze_articles(articles_df)
        bronze_companies = create_bronze_companies(companies_df)
        
        # ============ SILVER LAYER ============
        print("\n" + "="*80)
        print("[SILVER LAYER - Cleaned & Validated]")
        print("="*80)
        silver_articles = create_silver_articles(bronze_articles, bronze_companies, spark)

        # ============ GOLD LAYER (star schema) ============
        print("\n" + "="*80)
        print("[GOLD LAYER - Star Schema]")
        print("="*80)
        fact_arr_obs = create_fact_arr_observations(silver_articles)
        gold_dim_company = create_gold_dim_company(bronze_companies)
        gold_arr_latest = create_gold_arr_latest(fact_arr_obs)
        gold_ai_articles = create_gold_ai_articles_enriched(silver_articles)
        unmatched_companies = create_unmatched_companies_report(bronze_articles, bronze_companies, spark)
        invalid_records = create_invalid_records_report(bronze_articles)
        
        # ============ EXPORTS ============
        print("\n" + "="*80)
        print("[EXPORTING DATA BY LAYER]")
        print("="*80)
        
        # Bronze layer exports (raw data with parsing)
        print("\n[BRONZE Exports]")
        bronze_exports = {
            "bronze_articles.csv": bronze_articles,
            "bronze_companies.csv": bronze_companies,
        }
        for filename, df in bronze_exports.items():
            output_path = os.path.join(layer_dirs["bronze"], filename)
            rows = write_dataframe_to_csv(df, output_path)
            print(f"  ✓ {filename}: {rows} rows")
        
        # Silver layer exports (cleaned & validated data with matching)
        print("\n[SILVER Exports]")
        silver_exports = {
            "silver_articles.csv": silver_articles,
        }
        for filename, df in silver_exports.items():
            output_path = os.path.join(layer_dirs["silver"], filename)
            rows = write_dataframe_to_csv(df, output_path)
            print(f"  ✓ {filename}: {rows} rows")
        
        # Gold layer exports (analytical & star schema)
        print("\n[GOLD Exports]")
        
        # Export standard gold tables
        gold_exports = {
            "fact_arr_observations.csv": fact_arr_obs,
            "dim_company.csv": gold_dim_company,
            "gold_arr_latest.csv": gold_arr_latest,
            "unmatched_companies.csv": unmatched_companies,
            "invalid_records.csv": invalid_records,
        }
        for filename, df in gold_exports.items():
            output_path = os.path.join(layer_dirs["gold"], filename)
            rows = write_dataframe_to_csv(df, output_path)
            print(f"  ✓ {filename}: {rows} rows")
        
        # Enrich AI articles with embeddings and semantic similarity
        gold_ai_articles_with_embeddings = enrich_with_semantic_search(gold_ai_articles)
        
        # Export enriched AI articles (Pandas DataFrame, not Spark).
        # Named per the assignment's required deliverable: ai_articles_enriched.csv
        output_path = os.path.join(layer_dirs["gold"], "ai_articles_enriched.csv")
        rows = write_pandas_dataframe_to_csv(gold_ai_articles_with_embeddings, output_path)
        print(f"  [OK] ai_articles_enriched.csv: {rows} rows (with embeddings & semantic links)")
        
        # ============ PIPELINE SUCCESS ============
        print("\n" + "="*80)
        print(">>> PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"\nOutput structure: {os.path.abspath(output_dir)}/")
        print("   bronze/")
        print("     bronze_articles.csv")
        print("     bronze_companies.csv")
        print("   silver/")
        print("     silver_articles.csv")
        print("   gold/")
        print("     fact_arr_observations.csv")
        print("     dim_company.csv")
        print("     gold_arr_latest.csv")
        print("     ai_articles_enriched.csv (with embeddings & similar articles)")
        print("     unmatched_companies.csv")
        print("     invalid_records.csv")
        print("\nVector Database: ./vector_db.duckdb")
        print("  - Stores embeddings and similarity metadata")
        
        spark.stop()
        return True
        
    except Exception as e:
        print(f"\n[ERROR] PIPELINE FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        spark.stop()
        return False


if __name__ == "__main__":
    # Determine paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tech_news_path = os.path.join(script_dir, "tech_news.csv")
    company_metadata_path = os.path.join(script_dir, "company_metadata.json")
    output_dir = os.path.join(script_dir, "data_outputs")
    
    # Run pipeline
    success = run_pipeline(tech_news_path, company_metadata_path, output_dir)
    sys.exit(0 if success else 1)
