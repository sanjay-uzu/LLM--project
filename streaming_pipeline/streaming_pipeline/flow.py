import datetime
from pathlib import Path
from typing import Optional

from bytewax.dataflow import Dataflow
from bytewax.inputs import Input
from bytewax.outputs import Output
from bytewax.testing import TestingInput
from pydantic import parse_obj_as

from streaming_pipeline.alpaca_batch import AlpacaNewsBatchInput
from streaming_pipeline.alpaca_stream import AlpacaNewsStreamInput
from streaming_pipeline.embeddings import EmbeddingModelSingleton
from streaming_pipeline.models import NewsArticle
from streaming_pipeline.qdrant import QdrantVectorOutput

def build(is_batch: bool = False,
          from_datetime: Optional[datetime.datetime] = None,
          to_datetime: Optional[datetime.datetime] = None,
          model_cache_dir: Optional[Path] = None,
          debug: bool = False) -> Dataflow:
    """Builds a dataflow pipeline for processing news articles.

    Args:
        is_batch (bool): If True, processes a batch of articles, otherwise processes a stream.
        from_datetime (datetime.datetime, optional): Start datetime for batch processing.
        to_datetime (datetime.datetime, optional): End datetime for batch processing.
        model_cache_dir (Path, optional): Directory to cache the embedding model.
        debug (bool): Enables debug mode.

    Returns:
        Dataflow: Configured dataflow pipeline.
    """
    model = EmbeddingModelSingleton(cache_dir=model_cache_dir)
    input_source = _build_input(is_batch, from_datetime, to_datetime, debug)

    flow = Dataflow()
    flow.input("input", input_source)
    flow.flat_map(lambda messages: parse_obj_as(List[NewsArticle], messages))
    flow.map(lambda article: article.to_document())
    flow.map(lambda document: document.compute_chunks(model))
    flow.map(lambda document: document.compute_embeddings(model))
    flow.output("output", _build_output(debug, model))

    return flow

def _build_input(is_batch: bool,
                 from_datetime: Optional[datetime.datetime],
                 to_datetime: Optional[datetime.datetime],
                 debug: bool) -> Input:
    """Constructs the appropriate input source for the dataflow.

    Args:
        is_batch (bool): Determines whether to use batch or streaming input.
        from_datetime (datetime.datetime, optional): Start datetime for batch input.
        to_datetime (datetime.datetime, optional): End datetime for batch input.
        debug (bool): If True, uses mocked input for testing.

    Returns:
        Input: Configured input source for the dataflow.
    """
    if debug:
        return TestingInput(mocked.financial_news)
    elif is_batch:
        assert from_datetime and to_datetime, "Batch processing requires both from_datetime and to_datetime"
        return AlpacaNewsBatchInput(from_datetime=from_datetime, to_datetime=to_datetime, tickers=["*"])
    else:
        return AlpacaNewsStreamInput(tickers=["*"])

def _build_output(debug: bool, model: EmbeddingModelSingleton) -> Output:
    """Configures the output for the dataflow.

    Args:
        debug (bool): If True, outputs to an in-memory database for testing.
        model (EmbeddingModelSingleton): The embedding model used in the pipeline.

    Returns:
        Output: Configured output for the dataflow.
    """
    if debug:
        return QdrantVectorOutput(vector_size=model.max_input_length, client=QdrantClient(":memory:"))
    else:
        return QdrantVectorOutput(vector_size=model.max_input_length)

