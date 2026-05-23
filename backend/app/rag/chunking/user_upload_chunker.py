import re

from app.rag.chunking.simple_text_splitter import SimpleTextSplitter
from app.utils.text_utils import count_tokens_rough


class UserUploadChunker:
    def __init__(self, chunk_size: int = 1800, chunk_overlap: int = 220) -> None:
        self.splitter = SimpleTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def _normalize_text(self, text: str) -> str:
        lines = [line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
        cleaned: list[str] = []
        blank_streak = 0
        for line in lines:
            if not line.strip():
                blank_streak += 1
                if blank_streak <= 1:
                    cleaned.append("")
                continue
            blank_streak = 0
            cleaned.append(line.strip())
        return "\n".join(cleaned).strip()

    def chunk(self, markdown_text: str, base_metadata: dict) -> list[dict]:
        page_marker_pattern = re.compile(r"^<!--\s*page:\s*(\d+)\s*-->$", re.IGNORECASE)
        chunks: list[dict] = []
        current_page: int | None = None
        current_lines: list[str] = []

        def flush_current_page() -> None:
            text = self._normalize_text("\n".join(current_lines))
            if not text:
                return
            for split_text in self.splitter.split_text(text):
                cleaned = self._normalize_text(split_text)
                if not cleaned:
                    continue
                chunks.append(
                    {
                        "content": cleaned,
                        **base_metadata,
                        "chunk_index": len(chunks),
                        "section_title": None,
                        "token_count": count_tokens_rough(cleaned),
                        "page_number": current_page,
                        "page_start": current_page,
                        "page_end": current_page,
                        "page_source": "user_upload_page_marker" if current_page is not None else None,
                        "heading_path": [],
                        "chunk_type": "text",
                        "contains_table": "|" in cleaned,
                        "contains_image": False,
                    }
                )

        for raw_line in markdown_text.splitlines():
            marker = page_marker_pattern.match(raw_line.strip())
            if marker:
                flush_current_page()
                current_lines = []
                current_page = int(marker.group(1))
                continue
            current_lines.append(raw_line)

        flush_current_page()
        return chunks
