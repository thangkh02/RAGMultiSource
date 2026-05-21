from __future__ import annotations

from app.core.config import settings


class BGEEmbeddingService:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", max_length: int = 512) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self._tokenizer = None
        self._model = None
        self._openai_embeddings = None

    @property
    def openai_embeddings(self):
        if self._openai_embeddings is None:
            from langchain_openai import OpenAIEmbeddings

            self._openai_embeddings = OpenAIEmbeddings(
                model=settings.OPENAI_EMBEDDING_MODEL,
                api_key=settings.OPENAI_API_KEY,
            )
        return self._openai_embeddings

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self._tokenizer

    @property
    def model(self):
        if self._model is None:
            from transformers import AutoModel

            self._model = AutoModel.from_pretrained(self.model_name)
            self._model.eval()
        return self._model

    def _mean_pooling(self, model_output, attention_mask):
        import torch

        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def _encode_with_bge(self, texts: list[str]) -> list[list[float]]:
        import torch

        inputs = self.tokenizer(
            texts,
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

    def _encode(self, texts: list[str]) -> list[list[float]]:
        normalized_texts = [text.strip() for text in texts if text and text.strip()]
        if not normalized_texts:
            return []
        if settings.OPENAI_API_KEY:
            return self.openai_embeddings.embed_documents(normalized_texts)
        return self._encode_with_bge(normalized_texts)

    def embed_text(self, text: str) -> list[float]:
        embeddings = self._encode([text])
        if not embeddings:
            return []
        return embeddings[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts)
