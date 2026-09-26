import json
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, BooleanType

def read_tech_news_csv(spark, path):
    """Read tech_news.csv and return DataFrame"""
    df = spark.read \
        .option("header", "true") \
        .option("inferSchema", "false") \
        .csv(path)
    
    print(f"[OK] Loaded tech_news.csv: {df.count()} rows")
    return df


def read_company_metadata_json(spark, path):
    """Read company_metadata.json and convert to DataFrame"""
    with open(path, 'r') as f:
        metadata = json.load(f)
    
    # Convert nested JSON to list of records
    records = []
    for company_name, company_info in metadata.items():
        record = {
            'company_name': company_name,
            'founded_year': company_info.get('founded_year'),
            'headquarters': company_info.get('headquarters'),
            'employee_count': company_info.get('employee_count'),
            'industry': company_info.get('industry'),
            'is_public': company_info.get('is_public'),
            'stock_ticker': company_info.get('stock_ticker')
        }
        records.append(record)
    
    # Create DataFrame from records
    df = spark.createDataFrame(records, schema=StructType([
        StructField("company_name", StringType(), True),
        StructField("founded_year", IntegerType(), True),
        StructField("headquarters", StringType(), True),
        StructField("employee_count", IntegerType(), True),
        StructField("industry", StringType(), True),
        StructField("is_public", BooleanType(), True),
        StructField("stock_ticker", StringType(), True)
    ]))
    
    print(f"[OK] Loaded company_metadata.json: {df.count()} companies")
    return df
