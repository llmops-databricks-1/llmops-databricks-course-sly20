import yaml
import os


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
    cfg["parsed_table"] = f"{cat}.{sch}.parsed_documents"
    cfg["chunks_table"] = f"{cat}.{sch}.document_chunks"
    cfg["chunks_index"] = f"{cat}.{sch}.document_chunks_index"

    return cfg
