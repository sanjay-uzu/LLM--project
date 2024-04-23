import logging
from pathlib import Path
from typing import Optional, Tuple

import comet_ml
from datasets import Dataset
from peft import PeftConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EvalPrediction,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from trl import SFTTrainer

from training_pipeline import constants, metrics, models
from training_pipeline.configs import TrainingConfig
from training_pipeline.data import qa

logger = logging.getLogger(__name__)


class BestModelToModelRegistryCallback(TrainerCallback):
    def __init__(self, model_id: str):
        self._model_id = model_id

    @property
    def model_name(self) -> str:
        return f"financial_assistant/{self._model_id}"

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        best_model_checkpoint = state.best_model_checkpoint
        has_best_model_checkpoint = best_model_checkpoint is not None
        if has_best_model_checkpoint:
            best_model_checkpoint = Path(best_model_checkpoint)
            logger.info(
                f"Logging best model from {best_model_checkpoint} to the model registry..."
            )

            self.to_model_registry(best_model_checkpoint)
        else:
            logger.warning(
                "No best model checkpoint found. Skipping logging it to the model registry..."
            )

    def to_model_registry(self, checkpoint_dir: Path):
        checkpoint_dir = checkpoint_dir.resolve()

        assert (
            checkpoint_dir.exists()
        ), f"Checkpoint directory {checkpoint_dir} does not exist"

        # Get the stale experiment from the global context to grab the API key and experiment ID.
        stale_experiment = comet_ml.get_global_experiment()
        # Resume the expriment using its API key and experiment ID.
        experiment = comet_ml.ExistingExperiment(
            api_key=stale_experiment.api_key, experiment_key=stale_experiment.id
        )
        logger.info(f"Starting logging model checkpoint @ {self.model_name}")
        experiment.log_model(self.model_name, str(checkpoint_dir))
        experiment.end()
        logger.info(f"Finished logging model checkpoint @ {self.model_name}")


class TrainingAPI:
    def __init__(
        self,
        root_dataset_dir: Path,
        model_id: str,
        template_name: str,
        training_arguments: TrainingArguments,
        name: str = "training-api",
        max_seq_length: int = 1024,
        model_cache_dir: Path = constants.CACHE_DIR,
    ):
        self._root_dataset_dir = root_dataset_dir
        self._model_id = model_id
        self._template_name = template_name
        self._training_arguments = training_arguments
        self._name = name
        self._max_seq_length = max_seq_length
        self._model_cache_dir = model_cache_dir

        self._training_dataset, self._validation_dataset = self.load_data()
        self._model, self._tokenizer, self._peft_config = self.load_model()

    @classmethod
    def from_config(
        cls,
        config: TrainingConfig,
        root_dataset_dir: Path,
        model_cache_dir: Optional[Path] = None,
    ):
        return cls(
            root_dataset_dir=root_dataset_dir,
            model_id=config.model["id"],
            template_name=config.model["template"],
            training_arguments=config.training,
            max_seq_length=config.model["max_seq_length"],
            model_cache_dir=model_cache_dir,
        )

    def load_data(self) -> Tuple[Dataset, Dataset]:
        """
        Loads the training and validation datasets.

        Returns:
            Tuple[Dataset, Dataset]: A tuple containing the training and validation datasets.
        """

        logger.info(f"Loading QA datasets from {self._root_dataset_dir=}")

        training_dataset = qa.FinanceDataset(
            data_path=self._root_dataset_dir / "training_data.json",
            template=self._template_name,
            scope=constants.Scope.TRAINING,
        ).to_huggingface()
        validation_dataset = qa.FinanceDataset(
            data_path=self._root_dataset_dir / "testing_data.json",
            template=self._template_name,
            scope=constants.Scope.TRAINING,
        ).to_huggingface()

        logger.info(f"Training dataset size: {len(training_dataset)}")
        logger.info(f"Validation dataset size: {len(validation_dataset)}")

        return training_dataset, validation_dataset

    def load_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer, PeftConfig]:
        logger.info(f"Loading model using {self._model_id=}")
        model, tokenizer, peft_config = models.build_qlora_model(
            pretrained_model_name_or_path=self._model_id,
            gradient_checkpointing=True,
            cache_dir=self._model_cache_dir,
        )

        return model, tokenizer, peft_config

    def train(self) -> SFTTrainer:
        logger.info("Training model...")

        trainer = SFTTrainer(
            model=self._model,
            train_dataset=self._training_dataset,
            eval_dataset=self._validation_dataset,
            peft_config=self._peft_config,
            dataset_text_field="prompt",
            max_seq_length=self._max_seq_length,
            tokenizer=self._tokenizer,
            args=self._training_arguments,
            packing=True,
            compute_metrics=self.compute_metrics,
            callbacks=[BestModelToModelRegistryCallback(model_id=self._model_id)],
        )
        trainer.train()

        return trainer

    def compute_metrics(self, eval_pred: EvalPrediction):
        return {"perplexity": metrics.compute_perplexity(eval_pred.predictions)}
