# Databricks notebook source
# ingest data from sharepoint - nned prividge to get the sharepoint connection
# Documentation:  ADF SharePoint Online connector: https://learn.microsoft.com/en-us/azure/data-factory/connector-sharepoint-online-list?tabs=data-factory  (URL of a page that displays a list of files in a library)
# data engineer work
# for now, manual ingest in a Volume
from datetime import datetime

from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql.types import ArrayType, LongType, StringType, StructField, StructType

from helper import load_config, setup_logger, logger

setup_logger()

cfg = load_config(env="dev")

CATALOG = cfg["catalog"]
SCHEMA = cfg["schema"]
VOLUME = cfg["volume"]
DOC_PATH= f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"


logger.info(f"Catalog:     {CATALOG}")
logger.info(f"Schema:      {SCHEMA}")
logger.info(f"Volume:      {DOC_PATH}")


files = dbutils.fs.ls(DOC_PATH)
for f in files:
    print(f"{f.name:60s} {f.size:>10} bytes")
print(f"\nTotal: {len(files)} items")

# here should not be much since pdf come from the volume.
# if from SP, need to check ingestion pipeline and connection + priviledge

# OVERALL LOGIC
# 1- Read binary PDF files and parse with new db parsing capabilities: ai_parse_document
# 2- Extract text and metadata
# 3- Store in Delta Lake for chunking and vectorization
# 4- chunking
# 5- vectorization: create and store vector in vector database (we can use delta lake for that as well)
# 6- RAG Pipeline: retrieval + generation

# COMMAND ----------
PARSED_TABLE = cfg["parsed_table"]
PARSED_TABLE_LOCATION = cfg["parsed_table_location"]

logger.info(f"Parsed table: {PARSED_TABLE}")


df = spark.sql(f"""
CREATE OR REPLACE TABLE {PARSED_TABLE} AS
WITH parsed_docs AS (
  SELECT
    path,
    ai_parse_document(content) AS parsed
  FROM READ_FILES(
    '{DOC_PATH}/*.pdf',
    format => 'binaryFile'
  )
)
SELECT
  path,
  regexp_extract(path, '[^/]+$') AS filename,
  parsed,
  try_cast(parsed:error_status AS STRING) AS error_status,
  concat_ws('\\n\\n',
    transform(
      try_cast(parsed:document:elements AS ARRAY<VARIANT>),
      element -> try_cast(element:content AS STRING)
    )
  ) AS full_text,
  try_cast(parsed:document:pages AS ARRAY<VARIANT>) AS pages,
  size(try_cast(parsed:document:pages AS ARRAY<VARIANT>)) AS num_pages,
  current_timestamp() AS ingested_at
FROM parsed_docs
""")
display(df)
# COMMAND ----------
LLM_ENDPOINT = cfg["llm_endpoint"]
EMBEDDING_ENDPOINT = cfg["embedding_endpoint"]
WAREHOUSE_ID = cfg["warehouse_id"]
VS_ENDPOINT_NAME = cfg["vector_search_endpoint"]
GENIE_SPACE_ID = cfg["genie_space_id"]

CHUNKS_TABLE = cfg["chunks_table"]
CHUNKS_INDEX = cfg["chunks_index"]

logger.info(f"LLM:         {LLM_ENDPOINT}")
logger.info(f"Embedding:   {EMBEDDING_ENDPOINT}")
logger.info(f"VS Endpoint: {VS_ENDPOINT_NAME}")
