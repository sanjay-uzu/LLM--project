import logging
import os
import subprocess
from typing import Callable, Dict, List

import psutil
import torch

logger = logging.getLogger(__name__)


def log_available_gpu_memory():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            memory_info = subprocess.check_output(
                f"nvidia-smi -i {i} --query-gpu=memory.free --format=csv,nounits,noheader",
                shell=True,
            )
            memory_info = str(memory_info).split("\\")[0][2:]

            logger.info(f"GPU {i} memory available: {memory_info} MiB")
    else:
        logger.info("No GPUs available")


def log_available_ram():
    memory_info = psutil.virtual_memory()

    # convert bytes to GB
    logger.info(f"Available RAM: {memory_info.available / (1024.0 ** 3):.2f} GB")


def log_files_and_subdirs(directory_path: str):
    # Check if the directory exists
    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        for dirpath, dirnames, filenames in os.walk(directory_path):
            logger.info(f"Directory: {dirpath}")
            for filename in filenames:
                logger.info(f"File: {os.path.join(dirpath, filename)}")
            for dirname in dirnames:
                logger.info(f"Sub-directory: {os.path.join(dirpath, dirname)}")
    else:
        logger.info(f"The directory '{directory_path}' does not exist")


class MockedPipeline:
    task: str = "text-generation"

    def __init__(self, f: Callable[[str], str]):
        self.f = f

    def __call__(self, prompt: str) -> List[Dict[str, str]]:
        result = self.f(prompt)

        return [{"generated_text": f"{prompt}{result}"}]
