from loguru import logger
import yaml
import os
import sys


def load_config(env="dev", config_filename="config.yml"):
    """
    Load configuration from config.yml 

    Args:
        env: Environment key to load (default: "dev")
        config_filename: Name of the config file (default: "config.yml")

    Returns:
        dict with all config values plus derived paths:
            - catalog, schema, volume, llm_endpoint, embedding_endpoint
            - warehouse_id, vector_search_endpoint, genie_space_id
            - volume_path, parsed_table, chunks_table, chunks_index
    """

    config_path = "../project_config.yml"

    with open(config_path) as f:
        cfg = yaml.safe_load(f)[env]

    # Add derived paths
    cat = cfg["catalog"]
    sch = cfg["schema"]
    vol = cfg["volume"]
    base = cfg["storage_base"]
    cfg["parsed_table"] = f"{cat}.{sch}.parsed_documents"
    cfg["chunks_table"] = f"{cat}.{sch}.document_chunks"
    cfg["chunks_index"] = f"{cat}.{sch}.document_chunks_index"
    cfg["parsed_table_location"] = f"{base}/parsed_documents"
    cfg["chunks_table_location"] = f"{base}/document_chunks"
    
    return cfg

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