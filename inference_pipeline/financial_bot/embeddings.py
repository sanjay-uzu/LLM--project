import logging
import traceback
from typing import Optional, Union

import numpy as np
from transformers import AutoModel, AutoTokenizer

from financial_bot import constants
from financial_bot.base import SingletonMeta

logger = logging.getLogger(__name__)


class EmbeddingModelSingleton(metaclass=SingletonMeta):
    def __init__(
        self,
        model_id: str = constants.EMBEDDING_MODEL_ID,
        max_input_length: int = constants.EMBEDDING_MODEL_MAX_INPUT_LENGTH,
        device: str = "cuda:0",
        cache_dir: Optional[str] = None,
    ):
        self._model_id = model_id
        self._device = device
        self._max_input_length = max_input_length

        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModel.from_pretrained(
            model_id,
            cache_dir=str(cache_dir) if cache_dir else None,
        ).to(self._device)
        self._model.eval()

    @property
    def max_input_length(self) -> int:
        return self._max_input_length

    @property
    def tokenizer(self) -> AutoTokenizer:
        return self._tokenizer

    def __call__(
        self, input_text: str, to_list: bool = True
    ) -> Union[np.ndarray, list]:
        try:
            tokenized_text = self._tokenizer(
                input_text,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self._max_input_length,
            ).to(self._device)
        except Exception:
            logger.error(traceback.format_exc())
            logger.error(f"Error tokenizing the following input text: {input_text}")

            return [] if to_list else np.array([])

        try:
            result = self._model(**tokenized_text)
        except Exception:
            logger.error(traceback.format_exc())
            logger.error(
                f"Error generating embeddings for the following model_id: {self._model_id} and input text: {input_text}"
            )

            return [] if to_list else np.array([])

        embeddings = result.last_hidden_state[:, 0, :].cpu().detach().numpy()
        if to_list:
            embeddings = embeddings.flatten().tolist()

        return embeddings
