import re

from app.rag.chunking.chunk_utils import extract_heading
from app.rag.chunking.simple_text_splitter import SimpleTextSplitter
from app.utils.text_utils import count_tokens_rough


class MarkdownChunker:
    def __init__(self, chunk_size: int = 5000, chunk_overlap: int = 500) -> None:
        self.splitter = SimpleTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n## ", "\n\n", "\n- ", "\n", ".", " ", ""],
        )

    def _normalize_markdown_chunk(self, text: str) -> str:
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

    def _flush_section(
        self,
        chunks: list[dict],
        section_lines: list[str],
        base_metadata: dict,
        current_heading: str | None,
        heading_path: list[str],
        current_page: int | None,
        current_page_source: str | None,
    ) -> None:
        if not section_lines:
            return

        section_text = self._normalize_markdown_chunk("\n".join(section_lines))
        if not section_text:
            return

        split_chunks = self.splitter.split_text(section_text)
        for split_index, split_text in enumerate(split_chunks):
            cleaned = self._normalize_markdown_chunk(split_text)
            if not cleaned:
                continue
            if split_index > 0 and current_heading and not cleaned.startswith("#"):
                cleaned = f"## {current_heading}\n\n{cleaned}"
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
                    "page_source": current_page_source,
                    "heading_path": heading_path.copy(),
                    "chunk_type": "text",
                    "contains_table": "bảng " in cleaned.lower(),
                    "contains_image": False,
                }
            )

    def chunk(self, markdown_text: str, base_metadata: dict) -> list[dict]:
        lines = markdown_text.splitlines()
        chunks: list[dict] = []
        section_lines: list[str] = []
        current_heading = None
        heading_stack: list[tuple[int, str]] = []
        heading_path: list[str] = []
        current_page = None
        current_page_source = None
        page_marker_pattern = re.compile(r"^<!--\s*page:\s*(\d+)\s*-->$", re.IGNORECASE)
        page_source_pattern = re.compile(r"^<!--\s*page_source:\s*(?P<source>[^>]+?)\s*-->$", re.IGNORECASE)
        heading_pattern = re.compile(r"^(#{1,6})\s+(.*)$")

        def heading_level(line: str) -> int | None:
            match = heading_pattern.match(line.strip())
            if not match:
                return None
            return len(match.group(1))

        for line in lines:
            page_source = page_source_pattern.match(line.strip())
            if page_source:
                current_page_source = page_source.group("source").strip()
                continue

            page_marker = page_marker_pattern.match(line.strip())
            if page_marker:
                page_number = int(page_marker.group(1))
                self._flush_section(
                    chunks,
                    section_lines,
                    base_metadata,
                    current_heading,
                    heading_path,
                    current_page,
                    current_page_source,
                )
                section_lines = []
                current_page = page_number
                continue

            level = heading_level(line)
            if level is not None:
                heading = extract_heading(line)
                if heading:
                    self._flush_section(
                        chunks,
                        section_lines,
                        base_metadata,
                        current_heading,
                        heading_path,
                        current_page,
                        current_page_source,
                    )
                    section_lines = []
                    current_heading = heading
                    heading_stack = [(existing_level, existing_heading) for existing_level, existing_heading in heading_stack if existing_level < level]
                    heading_stack.append((level, heading))
                    heading_path = [stack_heading for _, stack_heading in heading_stack]
                    section_lines.append(line)
                    continue

            section_lines.append(line)

            if len("\n".join(section_lines)) > 12000:
                self._flush_section(
                    chunks,
                    section_lines,
                    base_metadata,
                    current_heading,
                    heading_path,
                    current_page,
                    current_page_source,
                )
                section_lines = []

        self._flush_section(chunks, section_lines, base_metadata, current_heading, heading_path, current_page, current_page_source)

        if chunks:
            return chunks

        text = self._normalize_markdown_chunk(markdown_text)
        if not text:
            return []

        for slice_text in self.splitter.split_text(text):
            cleaned = self._normalize_markdown_chunk(slice_text)
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
                    "page_source": None,
                    "heading_path": [],
                    "chunk_type": "text",
                    "contains_table": "bảng " in cleaned.lower(),
                    "contains_image": False,
                }
            )
        return chunks
