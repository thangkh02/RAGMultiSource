import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.rag.chunking.chunk_utils import extract_heading
from app.utils.text_utils import count_tokens_rough, normalize_whitespace


class MarkdownChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150) -> None:
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", ".", " ", ""],
            keep_separator=False,
        )

    def _flush_section(
        self,
        chunks: list[dict],
        section_lines: list[str],
        base_metadata: dict,
        current_heading: str | None,
        heading_path: list[str],
        current_page: int | None,
    ) -> None:
        if not section_lines:
            return

        section_text = normalize_whitespace("\n".join(section_lines))
        if not section_text:
            return

        split_chunks = self.splitter.split_text(section_text)
        for split_index, split_text in enumerate(split_chunks):
            cleaned = normalize_whitespace(split_text)
            if not cleaned:
                continue
            chunks.append(
                {
                    "content": cleaned,
                    **base_metadata,
                    "chunk_index": len(chunks),
                    "section_title": current_heading,
                    "token_count": count_tokens_rough(cleaned),
                    "page_number": current_page,
                    "page_start": current_page,
                    "page_end": current_page,
                    "heading_path": heading_path.copy(),
                    "chunk_type": "text",
                    "contains_table": False,
                    "contains_image": False,
                }
            )

    def chunk(self, markdown_text: str, base_metadata: dict) -> list[dict]:
        lines = markdown_text.splitlines()
        chunks: list[dict] = []
        section_lines: list[str] = []
        current_heading = None
        heading_path: list[str] = []
        current_page = None
        page_marker_pattern = re.compile(r"^<!--\s*page:\s*(\d+)\s*-->$", re.IGNORECASE)
        heading_pattern = re.compile(r"^(#{1,6})\s+(.*)$")

        def heading_level(line: str) -> int | None:
            match = heading_pattern.match(line.strip())
            if not match:
                return None
            return len(match.group(1))

        for line in lines:
            page_marker = page_marker_pattern.match(line.strip())
            if page_marker:
                page_number = int(page_marker.group(1))
                self._flush_section(chunks, section_lines, base_metadata, current_heading, heading_path, current_page)
                section_lines = []
                current_page = page_number
                continue

            level = heading_level(line)
            if level is not None:
                heading = extract_heading(line)
                if heading:
                    self._flush_section(chunks, section_lines, base_metadata, current_heading, heading_path, current_page)
                    section_lines = []
                    current_heading = heading
                    heading_path[:] = heading_path[: max(level - 1, 0)]
                    heading_path.append(heading)
                    section_lines.append(line)
                    continue

            heading = extract_heading(line)
            if heading:
                self._flush_section(chunks, section_lines, base_metadata, current_heading, heading_path, current_page)
                section_lines = []
                current_heading = heading
                section_lines.append(line)
            else:
                section_lines.append(line)

            if len("\n".join(section_lines)) > 2500:
                self._flush_section(chunks, section_lines, base_metadata, current_heading, heading_path, current_page)
                section_lines = []

        self._flush_section(chunks, section_lines, base_metadata, current_heading, heading_path, current_page)

        if chunks:
            return chunks

        text = normalize_whitespace(markdown_text)
        if not text:
            return []

        for slice_text in self.splitter.split_text(text):
            cleaned = normalize_whitespace(slice_text)
            if not cleaned:
                continue
            chunks.append(
                {
                    "content": cleaned,
                    **base_metadata,
                    "chunk_index": len(chunks),
                    "section_title": None,
                    "token_count": count_tokens_rough(cleaned),
                    "page_number": None,
                    "page_start": None,
                    "page_end": None,
                    "heading_path": [],
                    "chunk_type": "text",
                    "contains_table": False,
                    "contains_image": False,    
                }
            )
        return chunks
