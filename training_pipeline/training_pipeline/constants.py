from enum import Enum
from pathlib import Path

class ExecutionScope(Enum):
    TRAIN = "training"
    VALIDATE = "validation"
    TEST = "test"
    RUN = "inference"

MODEL_CACHE_DIRECTORY = Path.home() / ".cache" / "model-training"
