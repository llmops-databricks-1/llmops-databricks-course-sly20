# Databricks notebook source
# ingest data from sharepoint - nned prividge to get the sharepoint connection
# Documentation: https://learn.microsoft.com/en-us/azure/databricks/data/data-sources/azure/azure-sharepoint-online?tabs=python
# for now, manual ingest in a Volume
from datetime import datetime

from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql.types import ArrayType, LongType, StringType, StructField, StructType

from helper import load_config

cfg = load_config(env="dev")

doc_path= f"/Volumes/{cfg['catalog']}/{cfg['schema']}/{cfg['volume']}"
files = dbutils.fs.ls(doc_path)
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
