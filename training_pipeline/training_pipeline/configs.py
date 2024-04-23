from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from transformers import TrainingArguments

from training_pipeline.data.utils import read_yaml

@dataclass
class ModelTrainingConfig:
    training: TrainingArguments
    model: Dict[str, Any]

    @classmethod
    def from_yaml_file(cls, config_file_path: Path, output_directory: Path) -> "ModelTrainingConfig":
        config = read_yaml(config_file_path)
        config['training'] = cls.convert_to_training_args(config['training'], output_directory)
        return cls(**config)

    @classmethod
    def convert_to_training_args(cls, training_config: Dict, output_dir: Path) -> TrainingArguments:
        return TrainingArguments(
            output_dir=str(output_dir), **training_config
        )

@dataclass
class ModelInferenceConfig:
    model: Dict[str, Any]
    peft_model: Dict[str, Any]
    settings: Dict[str, Any]
    dataset: Dict[str, str]

    @classmethod
    def load_from_yaml(cls, config_file_path: Path) -> "ModelInferenceConfig":
        config = read_yaml(config_file_path)
        return cls(**config)
