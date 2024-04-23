import logging
import os
from pathlib import Path

import comet_llm
from datasets import Dataset
from peft import PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from training_pipeline import constants, models
from training_pipeline.configs import InferenceConfig
from training_pipeline.data import qa, utils
from training_pipeline.prompt_templates.prompter import get_llm_template

logger = logging.getLogger(__name__)


class InferenceAPI:
    def __init__(self,
                 peft_model_id: str,
                 model_id: str,
                 template_name: str,
                 root_dataset_dir: Path,
                 test_dataset_file: Path,
                 name: str = "inference-api",
                 max_new_tokens: int = 50,
                 temperature: float = 1.0,
                 model_cache_dir: Path = constants.CACHE_DIR,
                 debug: bool = False,
                 device: str = "cuda:0"):
        self._comet_project_name = self._get_comet_project_name()
        self._initialize_attributes(peft_model_id, model_id, template_name, root_dataset_dir,
                                    test_dataset_file, name, max_new_tokens, temperature,
                                    model_cache_dir, debug, device)

        self._model, self._tokenizer, self._peft_config = self.load_model()
        self._dataset = self.load_data() if root_dataset_dir else None

    def _get_comet_project_name(self):
        try:
            return os.environ["COMET_PROJECT_NAME"]
        except KeyError:
            logger.error("COMET_PROJECT_NAME environment variable not set.")
            raise

    def _initialize_attributes(self, peft_model_id, model_id, template_name, root_dataset_dir,
                               test_dataset_file, name, max_new_tokens, temperature,
                               model_cache_dir, debug, device):
        self._template_name = template_name
        self._prompt_template = get_llm_template(template_name)
        self._peft_model_id = peft_model_id
        self._model_id = model_id
        self._name = name
        self._root_dataset_dir = root_dataset_dir
        self._test_dataset_file = test_dataset_file
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._model_cache_dir = model_cache_dir
        self._debug = debug
        self._device = device

    def load_data(self) -> Dataset:
        dataset_path = self._root_dataset_dir / self._test_dataset_file
        max_samples = 3 if self._debug else None
        dataset = qa.FinanceDataset(data_path=dataset_path, template=self._template_name,
                                    scope=constants.Scope.INFERENCE, max_samples=max_samples).to_huggingface()
        logger.info(f"Loaded {len(dataset)} samples for inference")
        return dataset

    def load_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer, PeftConfig]:
        model, tokenizer, peft_config = models.build_qlora_model(pretrained_model_name_or_path=self._model_id,
                                                                 peft_pretrained_model_name_or_path=self._peft_model_id,
                                                                 gradient_checkpointing=False,
                                                                 cache_dir=self._model_cache_dir)
        model.to(self._device).eval()
        logger.info(f"Model loaded with ID: {self._model_id} on {self._device}")
        return model, tokenizer, peft_config

    def infer(self, infer_prompt: str, infer_payload: dict) -> str:
        logger.debug("Starting inference.")
        input_text = infer_prompt
        answer = models.prompt(model=self._model, tokenizer=self._tokenizer, input_text=input_text,
                               max_new_tokens=self._max_new_tokens, temperature=self._temperature,
                               device=self._device, return_only_answer=True)
        logger.debug(f"Inference completed: {answer[:50]}...")  # Log first 50 chars of answer
        return answer

    def infer_all(self, output_file: Optional[Path] = None) -> None:
        if not self._dataset:
            raise RuntimeError("Dataset not loaded. Cannot perform batch inference.")

        results = [(sample['prompt'], self.infer(sample['prompt'], sample['payload'])) for sample in self._dataset]
        if output_file:
            utils.write_json(results, output_file)
            logger.info(f"Results saved to {output_file}")
        else:
            logger.info("Batch inference completed without saving to file.")
