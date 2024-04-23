import logging
import logging.config
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv, find_dotenv

logger = logging.getLogger(__name__)

def setup_environment(logging_config_path: str = "logging.yaml", environment_file_path: str = ".env"):
    logger.info("Setting up logger...")
    try:
        setup_logger(config_path=logging_config_path)
    except FileNotFoundError:
        logger.warning(
            f"No logging configuration file found at {logging_config_path}. Defaulting to INFO level."
        )
        logging.basicConfig(level=logging.INFO)

    logger.info("Setting up environment variables...")
    env_file_path = environment_file_path or find_dotenv(raise_error_if_not_found=True, usecwd=True)
    if not load_dotenv(env_file_path, verbose=True):
        raise RuntimeError(f"Environment file not found at: {env_file_path}")

    os.environ["COMET_LOG_ASSETS"] = "True"
    os.environ["COMET_MODE"] = "ONLINE"

def setup_logger(config_path: str = "logging.yaml", logs_folder: str = "logs") -> None:
    log_directory = Path(config_path).parent / logs_folder
    log_directory.mkdir(parents=True, exist_ok=True)

    with open(config_path, 'rt') as file:
        config = yaml.safe_load(file.read())
    config["disable_existing_loggers"] = False
    logging.config.dictConfig(config)
