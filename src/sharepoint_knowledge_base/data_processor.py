"""
Sharepoint documents with ADF
   ↓ (download_and_store_documents)
PDFs in Volume + documents table
   ↓ (parse_pdfs_with_ai)
ai_parsed_docs_table (JSON)
   ↓ (process_chunks)
arxiv_chunks_table (clean text + metadata)
   ↓ (VectorSearchManager - separate class) (2.4 notebook)
Vector Search Index (embeddings)
"""

import json
import os
import re
import time
import uuid
from datetime import datetime

from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql import types as T
from pyspark.sql.functions import (
    col,
    concat_ws,
    current_timestamp,
    explode,
    udf,
)
from pyspark.sql.types import ArrayType, StringType, StructField, StructType
#from sklearn import base


from sharepoint_knowledge_base.config import ProjectConfig


def _extract_chunks_fn(parsed_content_json: str) -> list:
    """Extract text chunks from parsed_content JSON. Standalone for UDF serialization."""
    import json as _json  # import inside function so workers don't need the outer module
    parsed_dict = _json.loads(parsed_content_json)
    chunks = []
    for element in parsed_dict.get("document", {}).get("elements", []):
        if element.get("type") == "text":
            chunk_id = element.get("id", "")
            content = element.get("content", "")
            chunks.append((chunk_id, content))
    return chunks


def _clean_chunk_fn(text: str) -> str:
    """Clean and normalize chunk text. Standalone for UDF serialization."""
    import re as _re  # import inside function so workers don't need the outer module
    t = _re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    t = _re.sub(r"\s*\n\s*", " ", t)
    t = _re.sub(r"\s+", " ", t)
    return t.strip()



class DataProcessor:
    """
    DataProcessor handles the complete workflow of:
    - Downloading papers from sharepoint (ADF)
    - Storing paper metadata   (ADF)
    - Parsing PDFs with ai_parse_document
    - Extracting and cleaning text chunks
    - Saving chunks to Delta tables
    """

    def __init__(self, spark: SparkSession, config: ProjectConfig) -> None:
        """
        Initialize DataProcessor with Spark session and configuration.

        Args:
            spark: SparkSession instance
            config: ProjectConfig object with table configurations
        """
        self.spark = spark
        self.cfg = config
        self.catalog = config.catalog
        self.schema = config.schema
        self.volume = config.volume
        self.base = config.storage_base
        self.document_location = f"/Volumes/{self.catalog}/{self.schema}/{self.volume}"
        self.documents_metadata_table = f"{self.catalog}.{self.schema}.SP_documents_metadata"
        self.documents_metadata_table_location = f"{self.base}/SP_documents_metadata"
        self.parsed_table = f"{self.catalog}.{self.schema}.parsed_documents"
        self.parsed_table_location = f"{self.base}/parsed_documents"
        self.chunks_table = f"{self.catalog}.{self.schema}.document_chunks"
        self.chunks_table_location = f"{self.base}/document_chunks"
        self.chunks_index = f"{self.catalog}.{self.schema}.document_chunks_index" #not used
        self.end = time.strftime("%Y%m%d%H%M", time.gmtime())

    def _table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the catalog."""
        try:
            return self.spark.catalog.tableExists(table_name)
        except Exception:
            return False


    def _get_range_start(self) -> str:
        """
        Get start time range for document search. SD: should be in metadata of SP lib col
        If documents table exists, uses max(processed) as start.
        Otherwise, uses 3 days ago as start.

        Returns:
            start string in "YYYYMMDDHHMM" format
        """

        if self._table_exists(self.documents_metadata_table):
            result = self.spark.sql(f"""
                SELECT max(processed)
                FROM {self.documents_metadata_table}
            """).collect()
            start = str(result[0][0])
            logger.info(
                f"Found existing documents table. Starting from: {start}"
            )
        else:
            start = time.strftime(
                "%Y%m%d%H%M", time.gmtime(time.time() - 24 * 3600 * 3)
            )
            logger.info(
                f"No existing documents table. "
                f"Starting from 3 days ago: {start}"
            )
        return start
    
   ### not relevant for SP.  For SP, we should just read from volume and parse with ai_parse_document, since volume should be updated by ADF pipeline that moves files from SP to volume.
   ### important to have the correct metadata in the documents table, 
   ### title, author, published year, document category from SP should be extracted by ADF 

    def get_and_store_documents(
        self,
    ) -> list[dict] | None:
        """
        Get documents from volume and store metadata in documents table.  
        In production, this should be done by ADF pipeline that moves files from SP to volume, and create the delta table with metadata.

        Returns:
            List of paper metadata dictionaries if documents were downloaded,
            otherwise None
        """
        start = self._get_range_start()

        # Search for cocuments in volume

        from databricks.sdk import WorkspaceClient

        w = WorkspaceClient()

        # Download papers and collect metadata
        records = []

        for doc in w.files.list_directory_contents(self.document_location):
            try:
                # Collect metadata
                name_no_ext = doc.name.rsplit(".", 1)
                records.append(
                    {
                        "doc_id": str(uuid.uuid4()),
                        "title":  name_no_ext[0] if len(name_no_ext) > 0 else "",
                        "type": name_no_ext[1] if len(name_no_ext) > 1 else "",
                        "authors": [
                            "get_from_SP"
                        ],
                        "category": "get_from_SP",
                        "pdf_url": "get_from_SP",
                        "published": doc.last_modified, #modif time at the moment, otherwise from SP
                        "processed": int(self.end),  #should come from SP metadata
                        "volume_path": f"dbfs:{self.document_location}{doc.name}"
                    }
                )
            except Exception:
                logger.warning(
                    f"Document {doc.name} was not successfully processed."
                )
            # Avoid hitting API rate limits
            time.sleep(3)

        # Only process if we have records
        if len(records) == 0:
            logger.info("No new documents found.")
            return None

        logger.info(f"Found {len(records)} documents in {self.document_location}")

        # Create DataFrame and save to documents table
        schema = T.StructType(
            [
                T.StructField("doc_id", T.StringType(), False),  # UniqueId (GUID) coudld be used from SP?
                T.StructField("title", T.StringType(), True),
                T.StructField("type", T.StringType(), True),
                T.StructField("authors", T.ArrayType(T.StringType()), True),
                T.StructField("category", T.StringType(), True),
                T.StructField("pdf_url", T.StringType(), True),
                T.StructField("published", T.LongType(), True),
                T.StructField("processed", T.LongType(), True),
                T.StructField("volume_path", T.StringType(), True),
            ]
        )

        metadata_df = self.spark.createDataFrame(
            records, schema=schema).withColumn(
            "ingest_ts", current_timestamp()
        )
        
        logger.info(f"Metadata DataFrame: {len(metadata_df.columns)} columns: {metadata_df.columns}")
        logger.info(f"Schema: {metadata_df.dtypes}")

        # Create table if it doesn't exist
        table_exists = self._table_exists(self.documents_metadata_table)

        if not table_exists:
            # First run: create table with overwrite to ensure it's created
            logger.info(f"Creating new table {self.documents_metadata_table}")
            metadata_df.write.format("delta").mode("overwrite") \
                .option("overwriteSchema", "true") \
                .option("path", self.documents_metadata_table_location) \
                .saveAsTable(self.documents_metadata_table)
            logger.success(f"Created {self.documents_metadata_table} with {len(records)} records")
        else:
            # Subsequent runs: MERGE to avoid duplicates
            logger.info(f"Table exists, merging {len(records)} new records")
            metadata_df.createOrReplaceTempView("new_documents")
            self.spark.sql(f"""
                MERGE INTO {self.documents_metadata_table} target
                USING new_documents source
                ON target.doc_id = source.doc_id
                WHEN NOT MATCHED THEN INSERT (
                    doc_id, title, type, authors, category, pdf_url,
                    published, processed, volume_path, ingest_ts
                ) VALUES (
                    source.doc_id, source.title, source.type, source.authors,
                    source.category, source.pdf_url, source.published,
                    source.processed, source.volume_path, source.ingest_ts
                )
            """)
            logger.success(f"Merged {len(records)} records into {self.documents_metadata_table}")

        # Verify table was created successfully
        if not self._table_exists(self.documents_metadata_table):
            raise RuntimeError(
                f"Failed to create {self.documents_metadata_table}. "
                f"Check external location permissions for {self.documents_metadata_table_location}"
            )
        return records

    def parse_pdfs_with_ai(self) -> None:
        """
        Parse PDFs using ai_parse_document and store in ai_parsed_docs table.

        """

        # Guard: metadata table must exist before we can JOIN with it
        if not self._table_exists(self.documents_metadata_table):
            raise RuntimeError(
                f"Table {self.documents_metadata_table} does not exist. "
                f"Run get_and_store_documents() first."
            )


        logger.info(f"Parsing PDFs from {self.document_location}...")

        self.spark.sql(f"""
            CREATE OR REPLACE TABLE {self.parsed_table}
            LOCATION '{self.parsed_table_location}'
            AS
            WITH parsed_docs AS (
                SELECT
                    path,
                    ai_parse_document(content) AS parsed_content,
                    {self.end} AS processed
                FROM READ_FILES(
                    '{self.document_location}/*.pdf',
                    format => 'binaryFile'
                )
            )
            SELECT
                m.doc_id,
                m.volume_path,
                m.title,
                m.authors,
                m.category,
                m.published,
                p.processed,
                p.path,
                p.parsed_content,
                try_cast(p.parsed_content:error_status AS STRING) AS error_status,
                concat_ws('\\n\\n',
                    transform(
                        try_cast(p.parsed_content:document:elements AS ARRAY<VARIANT>),
                        element -> try_cast(element:content AS STRING)
                    )
                ) AS full_text,
                size(try_cast(p.parsed_content:document:pages AS ARRAY<VARIANT>)) AS num_pages
            FROM parsed_docs p
            JOIN {self.documents_metadata_table} m ON m.volume_path = p.path
        """)

        # Verify results
        count = self.spark.table(self.parsed_table).count()
        logger.success(
            f"Parsed {count} PDFs from {self.document_location} → {self.parsed_table}"
        )

    
    def process_chunks(self) -> None:
        """
        Process parsed documents to extract and clean chunks.
        Reads from ai_parsed_docs table and saves to chunks table.
        """
        # Guard: parsed table must exist
        if not self._table_exists(self.parsed_table):
            raise RuntimeError(
                f"Table {self.parsed_table} does not exist. "
                f"Run parse_pdfs_with_ai() first."
            )
        
        logger.info(
            f"Processing parsed documents from "
            f"{self.parsed_table} for end date {self.end}"
        )

        
        df = self.spark.table(self.parsed_table).where(
            f"processed = {self.end}"
        )

        row_count = df.count()
        logger.info(f"Parsed DataFrame: {row_count} rows, columns: {df.columns}")
        
        # issue with UDF and ModuleNotFoundError: No module named 'sharepoint_knowledge_base'
        # try to move teh udf outside the classe but did not help
        chunks_df = self.spark.sql(f"""
            WITH elements AS (
                SELECT
                    doc_id,
                    title,
                    authors,
                    category,
                    published,
                    processed,
                    explode(
                        filter(
                            transform(
                                try_cast(parsed_content:document:elements AS ARRAY<VARIANT>),
                                el -> struct(
                                    try_cast(el:id AS STRING) AS chunk_id,
                                    try_cast(el:type AS STRING) AS element_type,
                                    try_cast(el:content AS STRING) AS content
                                )
                            ),
                            el -> el.element_type = 'text'
                        )
                    ) AS element
                FROM {self.parsed_table}
                WHERE processed = {self.end}
            )
            SELECT
                doc_id,
                element.chunk_id AS chunk_id,
                -- Clean text: fix hyphenation, collapse newlines, collapse whitespace
                trim(regexp_replace(
                    regexp_replace(
                        regexp_replace(element.content, '(\\w)-\\s*\\n\\s*(\\w)', '$1$2'),
                        '\\s*\\n\\s*', ' '
                    ),
                    '\\s+', ' '
                )) AS text,
                concat(doc_id, '_', element.chunk_id) AS id,
                title,
                authors,
                category,
                published,
                processed AS ingested_at
            FROM elements
        """)
        
        logger.info(f"Chunks DataFrame: {chunks_df.count()} rows, columns: {chunks_df.columns}")

        table_exists = self._table_exists(self.chunks_table)
        write_mode = "append" if table_exists else "overwrite"
        logger.info(f"Writing chunks with mode='{write_mode}'")

        # Write to table
        chunks_df.write.mode("append") \
        .option("path", self.chunks_table_location) \
        .option("delta.enableChangeDataFeed", "true") \
        .saveAsTable(self.chunks_table)

        
        # Enable Change Data Feed
        self.spark.sql(f"""
            ALTER TABLE {self.chunks_table}
            SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
        """)

        chunk_count = self.spark.table(self.chunks_table).count()
        
        logger.success(f"Saved {chunk_count} chunks to {self.chunks_table}")

        # Enable Change Data Feed
         #self.spark.sql(f"""
          #   ALTER TABLE {self.chunks_table}
           #  SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
         #""")
         #logger.info(f"Change Data Feed enabled for {self.chunks_table}")

    def process_and_save(self) -> None:
        """
        Complete workflow: download papers, parse PDFs, and process chunks.
        """
        # Step 1: Get documents and store metadata
        records = self.get_and_store_documents()

        # Only continue if we have new papers
        if records is None:
            logger.info("No new papers to process. Exiting.")
            return

        # Step 2: Parse PDFs with ai_parse_document
        self.parse_pdfs_with_ai()
        logger.info("Parsed documents.")

        # Step 3: Process chunks
        self.process_chunks()
        logger.info("Processing complete!")