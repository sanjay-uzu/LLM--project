import logging
import traceback
from pathlib import Path

import numpy as np
from transformers import AutoModel, AutoTokenizer
import torch

from streaming_pipeline import constants
from streaming_pipeline.base import SingletonMeta

logger = logging.getLogger(__name__)

class EmbeddingModelSingleton(metaclass=SingletonMeta):
    """Singleton class for handling embeddings using a pre-trained transformer model.
    
    Attributes:
        model_id (str): Identifier for the pre-trained model.
        max_input_length (int): Max length of input text for tokenization.
        device (str): Computation device ('cpu' or 'gpu').
        tokenizer (AutoTokenizer): Tokenizer for the model.
        model (AutoModel): The pre-trained transformer model.
    """
    def __init__(self, model_id: str = constants.EMBEDDING_MODEL_ID,
                 max_input_length: int = constants.EMBEDDING_MODEL_MAX_INPUT_LENGTH,
                 device: str = constants.EMBEDDING_MODEL_DEVICE,
                 cache_dir: Path = None):
        self.model_id = model_id
        self.max_input_length = max_input_length
        self.device = device
        self.cache_dir = cache_dir

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id, cache_dir=str(cache_dir) if cache_dir else None).to(device)
        self.model.eval()  # Set the model to evaluation mode

    def __call__(self, input_text: str, to_list: bool = True):
        """Generates embeddings for input text.

        Args:
            input_text (str): The text to generate embeddings for.
            to_list (bool): Whether to return the result as a list. If False, returns a numpy array.

        Returns:
            Union[np.ndarray, list]: The computed embeddings.
        """
        try:
            # Tokenize text
            encoded_input = self.tokenizer(input_text, padding=True, truncation=True,
                                           return_tensors="pt", max_length=self.max_input_length).to(self.device)
            with torch.no_grad():  # Ensure no gradient is computed to save memory and computations
                outputs = self.model(**encoded_input)
            embeddings = outputs.last_hidden_state[:, 0, :].detach().numpy()  # Extract embeddings from model output
            return embeddings.flatten().tolist() if to_list else embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {traceback.format_exc()}")
            return [] if to_list else np.array([])

