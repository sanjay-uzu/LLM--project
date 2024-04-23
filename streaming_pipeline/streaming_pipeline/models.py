import hashlib
from datetime import datetime
from typing import List, Optional, Tuple

from pydantic import BaseModel
from unstructured.cleaners.core import (
    clean,
    clean_non_ascii_chars,
    replace_unicode_quotes,
)
from unstructured.partition.html import partition_html
from unstructured.staging.huggingface import chunk_by_attention_window

from streaming_pipeline.embeddings import EmbeddingModelSingleton


class NewsArticle(BaseModel):
    id: int
    headline: str
    summary: str
    author: str
    created_at: datetime
    updated_at: datetime
    url: Optional[str]
    content: str
    symbols: List[str]
    source: str

    def to_document(self) -> "Document":
        """
        Converts the news article to a Document object.

        Returns:
            Document: A Document object representing the news article.
        """

        document_id = hashlib.md5(self.content.encode()).hexdigest()
        document = Document(id=document_id)

        article_elements = partition_html(text=self.content)
        cleaned_content = clean_non_ascii_chars(
            replace_unicode_quotes(clean(" ".join([str(x) for x in article_elements])))
        )
        cleaned_headline = clean_non_ascii_chars(
            replace_unicode_quotes(clean(self.headline))
        )
        cleaned_summary = clean_non_ascii_chars(
            replace_unicode_quotes(clean(self.summary))
        )

        document.text = [cleaned_headline, cleaned_summary, cleaned_content]
        document.metadata["headline"] = cleaned_headline
        document.metadata["summary"] = cleaned_summary
        document.metadata["url"] = self.url
        document.metadata["symbols"] = self.symbols
        document.metadata["author"] = self.author
        document.metadata["created_at"] = self.created_at

        return document


class Document(BaseModel):
    id: str
    group_key: Optional[str] = None
    metadata: dict = {}
    text: list = []
    chunks: list = []
    embeddings: list = []

    def to_payloads(self) -> Tuple[List[str], List[dict]]:
        """
        Returns the payloads of the document.

        Returns:
            Tuple[List[str], List[dict]]: A tuple containing the IDs and payloads of the document.
        """

        payloads = []
        ids = []
        for chunk in self.chunks:
            payload = self.metadata
            payload.update({"text": chunk})
            # Create the chunk ID using the hash of the chunk to avoid storing duplicates.
            chunk_id = hashlib.md5(chunk.encode()).hexdigest()

            payloads.append(payload)
            ids.append(chunk_id)

        return ids, payloads

    def compute_chunks(self, model: EmbeddingModelSingleton) -> "Document":
        for item in self.text:
            chunked_item = chunk_by_attention_window(
                item, model.tokenizer, max_input_size=model.max_input_length
            )

            self.chunks.extend(chunked_item)

        return self

    def compute_embeddings(self, model: EmbeddingModelSingleton) -> "Document":
        for chunk in self.chunks:
            embedding = model(chunk, to_list=True)

            self.embeddings.append(embedding)

        return self
