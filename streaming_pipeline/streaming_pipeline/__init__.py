import logging
import logging.config
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

def setup_logging(default_path='logging.yaml', default_level=logging.INFO, env_key='LOG_CFG'):
    """Setup logging configuration

    Args:
        default_path (str): Default path to logging configuration file.
        default_level (int): Default logging level.
        env_key (str): Environment key for logging configuration file path.
    """
    path = os.getenv(env_key, default_path)
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)
        logging.warning(f'Failed to load configuration file. Using default configs')

def load_environment_variables(dotenv_path='.env'):
    """Load environment variables from a .env file

    Args:
        dotenv_path (str): Path to .env file.
    """
    load_dotenv(dotenv_path)
    logging.info(f'Environment variables loaded from {dotenv_path}')

if __name__ == "__main__":
    setup_logging()
    load_environment_variables()
