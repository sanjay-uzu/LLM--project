# Embedding model configuration for NLP processing
EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL_MAX_INPUT_LENGTH = 384  # Maximum number of tokens for the model input
EMBEDDING_MODEL_DEVICE = "cpu"  # Computation device: 'cpu' or 'gpu'

# Database configuration for storing and retrieving vectors
VECTOR_DB_OUTPUT_COLLECTION_NAME = "alpaca_financial_news"
