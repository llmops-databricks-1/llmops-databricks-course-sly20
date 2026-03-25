# Databricks notebook source
# ingest data from sharepoint - need prividge to get the sharepoint connection
# Documentation:  ADF SharePoint Online connector: https://learn.microsoft.com/en-us/azure/data-factory/connector-sharepoint-online-list?tabs=data-factory  (URL of a page that displays a list of files in a library)
# data engineer work - seems possible to get metadata from SP library cols
# for now, manual ingest in a Volume

from databricks.connect import DatabricksSession
from loguru import logger

import sharepoint_knowledge_base

from sharepoint_knowledge_base.config import get_env, load_config, logger, setup_logger
from sharepoint_knowledge_base.data_processor import DataProcessor

from databricks.vector_search.client import VectorSearchClient
from databricks.vector_search.reranker import DatabricksReranker

from sharepoint_knowledge_base.vector_search import VectorSearchManager


spark = DatabricksSession.builder.getOrCreate()
logger.info("✅ Using Databricks Connect Spark session")

setup_logger()

env = get_env(spark)
cfg = load_config("../project_config.yml", env)
catalog = cfg.catalog
schema = cfg.schema
volume = cfg.volume

logger.info(f"Catalog: {catalog}, Schema: {schema}, Volume: {volume}")

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
# PROCESSING DATA
processor = DataProcessor(spark=spark, config=cfg)

processor.process_and_save()

# COMMAND ----------
# VECTOR SEARCH
vs_manager = VectorSearchManager(
    config=cfg,
    endpoint_name=cfg.vector_search_endpoint,
    embedding_model=cfg.embedding_endpoint
)
logger.info(f"Vector Search Endpoint: {vs_manager.endpoint_name}")
logger.info(f"Embedding Model: {vs_manager.embedding_model}")
logger.info(f"Index Name: {vs_manager.index_name}")
# COMMAND ----------
vs_manager.create_endpoint_if_not_exists()
logger.success(f"Endpoint created: {vs_manager.endpoint_name}")
# COMMAND ----------
index = vs_manager.create_or_get_index()

logger.success(f"f Index Created: {vs_manager.index_name}")
logger.success(f"\n✓ Vector search setup complete!")
logger.info(f"  Index: {vs_manager.index_name}")
logger.info(f"  Source: {vs_manager.catalog}.{vs_manager.schema}.document_chunks")
logger.info(f"  Embedding Model: {vs_manager.embedding_model}")

# COMMAND ----------

def parse_vector_search_results(results):
    """Parse vector search results from array format to dict format.
    
    Args:
        results: Raw results from similarity_search()
        
    Returns:
        List of dictionaries with column names as keys
    """
    columns = [col['name'] for col in results.get('manifest', {}).get('columns', [])]
    data_array = results.get('result', {}).get('data_array', [])
    
    return [dict(zip(columns, row_data)) for row_data in data_array]



# Simple similarity search
query = "What are the best pratice in nitriding applied to extrusion"

results = index.similarity_search(
    query_text=query,
    columns=["text", "id", "title", "arxiv_id"],
    num_results=5
)

logger.info(f"Query: {query}\n")
logger.info("Top 5 Results:")
logger.info("=" * 80)

# Parse results using helper function
for i, row in enumerate(parse_vector_search_results(results), 1):
    logger.info(f"\n{i}. Paper: {row.get('title', 'N/A')}")
    logger.info(f"   arXiv ID: {row.get('arxiv_id', 'N/A')}")
    logger.info(f"   Chunk ID: {row.get('id', 'N/A')}")
    logger.info(f"   Text preview: {row.get('text', '')[:200]}...")
    logger.info(f"   Score: {row.get('score', 'N/A'):.4f}")
