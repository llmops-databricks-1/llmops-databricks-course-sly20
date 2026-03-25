import sys

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession
from loguru import logger



def load_config_old(env="dev", config_filename="project_config.yml"):
    """Load configuration from a YAML file.

    Args:
        env: Environment key to load (default: "dev").
        config_filename: Name of the config file (default: "project_config.yml").

    Returns:
        dict with all config values plus derived paths:
            - catalog, schema, volume, llm_endpoint, embedding_endpoint
            - warehouse_id, vector_search_endpoint, genie_space_id
            - parsed_table, parsed_table_location, chunks_table, chunks_index

    Notes:
        * Uses a path relative to this helper module to avoid dependency on the
          current working directory.
        * If the requested env is missing, a helpful KeyError is raised.
    """

    # Resolve config path relative to this module, so it works consistently
    # whether the notebook is executed from VS Code, Databricks, or other runtimes.
    base_dir = os.path.dirname(__file__)
    config_path = os.path.abspath(os.path.join(base_dir, "..", config_filename))

    logger.info(f"Loading config from {config_path} (env={env})")

    with open(config_path) as f:
        full_cfg = yaml.safe_load(f)

    if env not in full_cfg:
        raise KeyError(f"Environment '{env}' not found in {config_path}. Available: {list(full_cfg.keys())}")

    cfg = full_cfg[env]

    # Add derived paths
    cat = cfg["catalog"]
    sch = cfg["schema"]
    vol = cfg["volume"]
    base = cfg["storage_base"]
    cfg["documents_metadata_table"]= f"{cat}.{sch}.SP_documents_metadata"
    cfg["parsed_table"] = f"{cat}.{sch}.parsed_documents"
    cfg["parsed_table_location"] = f"{base}/parsed_documents"
    cfg["chunks_table"] = f"{cat}.{sch}.document_chunks"
    cfg["chunks_index"] = f"{cat}.{sch}.document_chunks_index"
    # Derived storage paths for managed/external Delta tables
    # NOTE: `parsed_table_location` is used by notebooks to specify the table LOCATION

    return cfg

class ProjectConfig(BaseModel):
    """Project configuration model."""

    catalog: str = Field(..., description="Unity Catalog name")
    db_schema: str = Field(..., description="Schema name", alias="schema")
    volume: str = Field(..., description="Volume name")
    storage_base: str = Field(..., description="Base path for storage tables (e.g., 'abfss://...')")
    llm_endpoint: str = Field(..., description="LLM endpoint name")
    embedding_endpoint: str = Field(..., description="Embedding endpoint name")
    warehouse_id: str = Field(..., description="Warehouse ID")
    vector_search_endpoint: str = Field(..., description="Vector search endpoint name")
    genie_space_id: str | None = Field(None, description="Genie space ID for MCP integration")
    system_prompt: str = Field(
        default="You are a helpful and structured AI assistant expert in aluminium extrusion. You helps users to understand aluminium extrusion and to solve issues by having a step-by-step scientifical and analytical approach. If you do not know, do not make it up.",
        description="System prompt for the agent"
    )
    
    model_config = {"populate_by_name": True}

    @classmethod
    def from_yaml(cls, config_path: str, env: str = "dev") -> "ProjectConfig":
        """Load configuration from YAML file.

        Args:
            config_path: Path to the YAML configuration file
            env: Environment name (dev, acc, prd)

        Returns:
            ProjectConfig instance
        """
        if env not in ["prd", "acc", "dev"]:
            raise ValueError(f"Invalid environment: {env}. Expected 'prd', 'acc', or 'dev'")

        with open(config_path) as f:
            config_data = yaml.safe_load(f)

        if env not in config_data:
            raise ValueError(f"Environment '{env}' not found in config file")

        return cls(**config_data[env])

    @property
    def schema(self) -> str:
        """Alias for db_schema for backward compatibility."""
        return self.db_schema
    
    @property
    def full_schema_name(self) -> str:
        """Get fully qualified schema name."""
        return f"{self.catalog}.{self.db_schema}"

    @property
    def full_volume_path(self) -> str:
        """Get fully qualified volume path."""
        return f"{self.catalog}.{self.schema}.{self.volume}"


class ModelConfig(BaseModel):
    """Model configuration."""

    temperature: float = Field(0.7, description="Model temperature")
    max_tokens: int = Field(2000, description="Maximum tokens")
    top_p: float = Field(0.95, description="Top-p sampling parameter")


class VectorSearchConfig(BaseModel):
    """Vector search configuration."""

    embedding_dimension: int = Field(1024, description="Embedding dimension")
    similarity_metric: str = Field("cosine", description="Similarity metric")
    num_results: int = Field(5, description="Number of results to return")


class ChunkingConfig(BaseModel):
    """Chunking configuration."""

    chunk_size: int = Field(512, description="Chunk size in tokens")
    chunk_overlap: int = Field(50, description="Overlap between chunks")
    separator: str = Field("\n\n", description="Separator for chunking")




def load_config(config_path: str = "project_config.yml", env: str = "dev") -> ProjectConfig:
    """Load project configuration.
    
    Args:
        config_path: Path to configuration file
        env: Environment name
        
    Returns:
        ProjectConfig instance
    """
    # Handle relative paths from notebooks
    if not Path(config_path).is_absolute():
        # Try to find config in parent directories
        current = Path.cwd()
        for _ in range(3):  # Search up to 3 levels
            candidate = current / config_path
            if candidate.exists():
                config_path = str(candidate)
                break
            current = current.parent
    
    return ProjectConfig.from_yaml(config_path, env)




def get_env(spark: SparkSession) -> str:
    """Get current environment from dbutils widget, falling back to ENV variable or 'dev'.

    Returns:
        Environment name (dev, acc, or prd)
    """
    try:
        dbutils = DBUtils(spark)
        return dbutils.widgets.get("env")
    except Exception:
        return "dev"


def setup_logger(level="INFO"):
    """
    Configure loguru logger with a consistent format.
    Call once at the start of a notebook. Safe to call multiple times.

    Args:
        level: Logging level (default: "INFO")

    Returns:
        loguru.logger instance
    """
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<7} | {message}",
        level=level,
        colorize=True,
    )
    return logger
