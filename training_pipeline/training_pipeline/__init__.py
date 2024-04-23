import os
from pathlib import Path

import yaml
from dotenv import load_dotenv, find_dotenv

def setup_environment(logging_config_path: str = "logging.yaml", environment_file_path: str = ".env"):
    env_file_path = environment_file_path or find_dotenv(raise_error_if_not_found=True, usecwd=True)
    if not load_dotenv(env_file_path, verbose=True):
        raise RuntimeError(f"Environment file not found at: {env_file_path}")

    os.environ["COMET_LOG_ASSETS"] = "True"
    os.environ["COMET_MODE"] = "ONLINE"


