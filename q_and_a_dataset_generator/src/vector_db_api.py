import os
from typing import Dict, List
from env import QDRANT_URL ,QDRANT_API_KEY
import numpy as np
import logging as logger
from transformers import AutoTokenizer, AutoModel
QDRANT_API_URL = QDRANT_URL 
QDRANT_API_KEY = QDRANT_API_KEY

# def fetch_relevant_news_from_db() -> List[Dict]:
#     """"""
#     from qdrant_client import QdrantClient

#     qdrant_client = QdrantClient(
#         url="https://ae87052d-4765-49f5-b462-d9b468a97976.eu-central-1-0.aws.cloud.qdrant.io:6333", 
#         api_key="<your-token>",
#     )


from qdrant_client import QdrantClient

def get_qdrant_client() -> QdrantClient:
    """"""
    qdrant_client = QdrantClient(
        url=QDRANT_API_URL, 
        api_key=QDRANT_API_KEY,
    )

    return qdrant_client

def init_collection(
    qdrant_client: QdrantClient,
    collection_name: str,
    vector_size: int,
    # schema: str = ''
) -> QdrantClient:
    """"""
    from qdrant_client.http.api_client import UnexpectedResponse
    from qdrant_client.http.models import Distance, VectorParams

    try: 
        qdrant_client.get_collection(collection_name=collection_name)

    except (UnexpectedResponse, ValueError):
        qdrant_client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            ),
            # schema=schema
    )

    return qdrant_client
def load_model():
    """Load the tokenizer and model for embedding generation."""
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    return tokenizer, model

# Load tokenizer and model once to reuse
def text_to_vector(text: str) -> np.ndarray:
    """Convert input text to vector using a pre-trained language model."""
    tokenizer, model = load_model()
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    outputs = model(**inputs)
    # Use the first token's output embeddings (CLS token for BERT-like models)
    vector = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy().flatten()
    return vector

def search_vectors(qdrant_client: QdrantClient, collection_name: str, query_text: str, top_k: int = 5) -> List[Dict]:
    """Search for top_k similar vectors in the given Qdrant collection."""
    try:
        query_vector = text_to_vector(query_text)
        results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_vector.tolist(),  # Ensure the vector is a list
             # Assuming 'top' is directly accepted by the search method if `SearchParams` is not used
            # Make sure payload selection is correct
            with_payload={"include": ['title', 'url', 'summary']}  # Adjust the parameter based on library documentation
        )
        return results
    except Exception as e:
        logger.error(f"Failed to search vectors: {str(e)}")
        return []