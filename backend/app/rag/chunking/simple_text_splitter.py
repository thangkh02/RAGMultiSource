class SimpleTextSplitter:
    def __init__(self, chunk_size: int, chunk_overlap: int = 0, separators: list[str] | None = None) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = min(chunk_overlap, max(chunk_size - 1, 0))
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def split_text(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            if end < len(text):
                split_at = self._find_split_point(text, start, end)
                if split_at > start:
                    end = split_at
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(text):
                break
            start = max(end - self.chunk_overlap, start + 1)
        return chunks

    def _find_split_point(self, text: str, start: int, end: int) -> int:
        window = text[start:end]
        for separator in self.separators:
            if separator == "":
                continue
            index = window.rfind(separator)
            if index > 0:
                return start + index + len(separator)
        return end
