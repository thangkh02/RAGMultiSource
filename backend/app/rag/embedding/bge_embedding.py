from __future__ import annotations

from functools import lru_cache

import torch
from transformers import AutoModel, AutoTokenizer


class BGEEmbeddingService:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", max_length: int = 512) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self._tokenizer = None
        self._model = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self._tokenizer

    @property
    def model(self):
        if self._model is None:
            self._model = AutoModel.from_pretrained(self.model_name)
            self._model.eval()
        return self._model

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def _encode(self, texts: list[str]) -> list[list[float]]:
        normalized_texts = [text.strip() for text in texts if text and text.strip()]
        if not normalized_texts:
            return []

        inputs = self.tokenizer(
            normalized_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        with torch.no_grad():
            model_output = self.model(**inputs)
            embeddings = self._mean_pooling(model_output, inputs["attention_mask"])
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().tolist()

    def embed_text(self, text: str) -> list[float]:
        embeddings = self._encode([text])
        if not embeddings:
            return []
        return embeddings[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts)
