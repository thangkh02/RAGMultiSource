from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from docx import Document as DocxDocument
from pypdf import PdfReader

try:
    from markitdown import MarkItDown
except ImportError as exc:  # pragma: no cover - handled at runtime
    MarkItDown = None  # type: ignore[assignment]
    _MARKITDOWN_IMPORT_ERROR = exc
else:
    _MARKITDOWN_IMPORT_ERROR = None


class MarkItDownMarkdownConverter:
    _page_marker_pattern = re.compile(r"^##\s*Page\s+(?P<page>\d+)\s*$", re.IGNORECASE)
    _numeric_heading_pattern = re.compile(r"^(?P<num>\d+(?:\.\d+)*)\.\s+(?P<title>.+?)\s*$")

    def __init__(self) -> None:
        self._converter = MarkItDown() if MarkItDown is not None else None

    def _normalize_line(self, line: str) -> str:
        return re.sub(r"\s+", " ", line).strip()

    def _normalize_whitespace(self, text: str) -> str:
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
            cleaned.append(line)
        return "\n".join(cleaned).strip()

    def _extract_markdown(self, input_file: Path) -> str:
        markdown_text = ""
        if self._converter is not None:
            result = self._converter.convert(str(input_file))
            markdown_text = getattr(result, "text_content", "") or getattr(result, "markdown", "")
        if not isinstance(markdown_text, str) or not markdown_text.strip():
            markdown_text = self._fallback_convert(input_file)
        if not isinstance(markdown_text, str) or not markdown_text.strip():
            raise ValueError(f"Could not extract markdown from {input_file}")
        return markdown_text

    def _fallback_convert_pdf(self, input_file: Path) -> str:
        reader = PdfReader(str(input_file))
        parts: list[str] = []
        for page_index, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            parts.append(f"## Page {page_index}\n\n{text.strip()}\n")
        return "\n".join(part for part in parts if part.strip()).strip()

    def _fallback_convert_docx(self, input_file: Path) -> str:
        doc = DocxDocument(str(input_file))
        parts: list[str] = []
        for paragraph in doc.paragraphs:
            text = paragraph.text.strip()
            if not text:
                continue
            if paragraph.style and paragraph.style.name:
                style_name = paragraph.style.name.lower()
                if style_name.startswith("heading"):
                    try:
                        level = int("".join(ch for ch in style_name if ch.isdigit()) or "1")
                    except ValueError:
                        level = 1
                    level = max(1, min(level, 6))
                    parts.append(f"{'#' * level} {text}")
                    continue
            parts.append(text)
        return "\n\n".join(parts).strip()

    def _fallback_convert(self, input_file: Path) -> str:
        suffix = input_file.suffix.lower()
        if suffix == ".pdf":
            return self._fallback_convert_pdf(input_file)
        if suffix == ".docx":
            return self._fallback_convert_docx(input_file)
        raise ValueError(f"Unsupported file type for fallback conversion: {suffix}")

    def _clean_none(self, markdown_text: str) -> str:
        return markdown_text

    def _clean_default(self, markdown_text: str) -> str:
        return self._normalize_whitespace(markdown_text)

    def _is_noise_line(self, line: str) -> bool:
        normalized = self._normalize_line(line)
        if not normalized:
            return True
        if normalized.isdigit():
            return True
        if self._page_marker_pattern.match(normalized):
            return True
        if re.fullmatch(r"[0-9]{1,4}", normalized):
            return True
        if "Tập " in normalized and "Số " in normalized and any(char.isdigit() for char in normalized):
            return True
        if "Jos.hueuni.edu.vn" in normalized:
            return True
        if "Tác giả liên hệ" in normalized or "Correspondence to" in normalized:
            return True
        if normalized.startswith("*Tác giả liên hệ") or normalized.startswith("*Correspondence to"):
            return True
        if normalized.startswith("(Received:") or normalized.startswith("(Ngày nhận bài:"):
            return True
        noise_prefixes = (
            "ISSN ",
            "DOI:",
            "Tập ",
            "Số ",
            "Vol.",
            "Volume ",
            "Tr.",
            "Nguyễn ",
            "Nguyen ",
            "Trường ",
            "Đại học ",
            "ĐH ",
            "Học viện ",
            "University ",
            "Faculty ",
            "Khoa ",
        )
        if any(normalized.startswith(prefix) for prefix in noise_prefixes):
            return True
        return False

    def _looks_like_author_line(self, line: str) -> bool:
        normalized = self._normalize_line(line)
        if not normalized:
            return False
        if "@" in normalized or "<" in normalized or ">" in normalized:
            return True
        if any(keyword in normalized for keyword in ("Tác giả liên hệ", "Correspondence to")):
            return True
        if normalized.startswith("*"):
            return True
        if any(keyword in normalized for keyword in ("Trường ", "Đại học ", "Học viện ", "University ", "Faculty ", "Khoa ", "Viện ")):
            return True
        words = [word for word in re.split(r"\s+", normalized) if word]
        if 2 <= len(words) <= 6:
            alpha_words = [word for word in words if any(ch.isalpha() for ch in word)]
            if len(alpha_words) == len(words):
                title_case_words = 0
                for word in alpha_words:
                    first = next((ch for ch in word if ch.isalpha()), "")
                    if first and first.isupper():
                        title_case_words += 1
                if title_case_words >= max(2, len(alpha_words) - 1):
                    return True
        return False

    def _looks_like_title_line(self, line: str) -> bool:
        normalized = self._normalize_line(line)
        if len(normalized) < 12:
            return False
        if self._is_noise_line(normalized):
            return False
        letters = [ch for ch in normalized if ch.isalpha()]
        if len(letters) < 5:
            return False
        upper_letters = sum(1 for ch in letters if ch.isupper())
        return upper_letters / len(letters) >= 0.55

    def _sentence_case(self, text: str) -> str:
        normalized = self._normalize_line(text).lower()
        if not normalized:
            return normalized
        return normalized[0].upper() + normalized[1:]

    def _collect_pages(self, markdown_text: str) -> list[tuple[int, list[str]]]:
        pages: list[tuple[int, list[str]]] = []
        current_page = 0
        current_lines: list[str] = []
        for raw_line in markdown_text.splitlines():
            marker = self._page_marker_pattern.match(raw_line.strip())
            if marker:
                if current_lines:
                    pages.append((current_page or 1, current_lines))
                current_page = int(marker.group("page"))
                current_lines = []
                continue
            current_lines.append(raw_line)
        if current_lines:
            pages.append((current_page or 1, current_lines))
        return pages

    def _collect_common_noise(self, pages: list[tuple[int, list[str]]]) -> set[str]:
        counter: Counter[str] = Counter()
        page_count = len(pages)
        if page_count == 0:
            return set()

        sample_window = 4
        for _, lines in pages:
            non_empty = [self._normalize_line(line) for line in lines if self._normalize_line(line)]
            for line in non_empty[:sample_window]:
                if line:
                    counter[line] += 1
            for line in non_empty[-sample_window:]:
                if line:
                    counter[line] += 1

        threshold = 2 if page_count < 4 else max(2, (page_count + 1) // 2)
        return {line for line, count in counter.items() if count >= threshold and len(line) <= 120}

    def _format_heading_line(self, line: str) -> str:
        normalized = self._normalize_line(line)
        match = self._numeric_heading_pattern.match(normalized)
        if not match:
            return normalized
        number = match.group("num")
        title = self._normalize_line(match.group("title"))
        depth = min(number.count(".") + 1, 6)
        if depth <= 1:
            depth = 2
        return f"{'#' * depth} {number}. {title}"

    def _clean_vi_scientific_paper(self, markdown_text: str) -> str:
        pages = self._collect_pages(markdown_text)
        if not pages:
            return self._normalize_whitespace(markdown_text)

        common_noise = self._collect_common_noise(pages)
        output: list[str] = []

        for page_number, lines in pages:
            cleaned_lines: list[str] = []
            for raw_line in lines:
                normalized = self._normalize_line(raw_line)
                if not normalized:
                    continue
                if normalized in common_noise:
                    continue
                if self._is_noise_line(normalized):
                    continue
                cleaned_lines.append(normalized)

            if not cleaned_lines:
                continue

            page_output: list[str] = [f"<!-- page: {page_number} -->"]
            if page_number == 1:
                front_matter_lines: list[str] = []
                body_lines: list[str] = []
                body_started = False
                for line in cleaned_lines:
                    if not body_started and (
                        line.startswith("Tóm tắt")
                        or line.startswith("Abstract")
                        or line.startswith("Từ khóa")
                        or line.startswith("Keywords")
                        or line.startswith("1.")
                        or line.startswith("2.")
                        or line.startswith("3.")
                        or line.startswith("4.")
                    ):
                        body_started = True
                    if body_started:
                        if self._looks_like_title_line(line):
                            continue
                        body_lines.append(line)
                        continue
                    if self._looks_like_author_line(line):
                        continue
                    front_matter_lines.append(line)

                title = self._sentence_case(" ".join(front_matter_lines)) if front_matter_lines else None
                if title:
                    page_output.append(f"# {title}")
                for line in body_lines:
                    page_output.append(self._format_heading_line(line))
            else:
                for line in cleaned_lines:
                    page_output.append(self._format_heading_line(line))

            output.append("\n".join(page_output).strip())

        return "\n\n".join(block for block in output if block).strip()

    def _apply_cleanup(self, markdown_text: str, cleanup_profile: str) -> str:
        if cleanup_profile == "none":
            return self._clean_none(markdown_text)
        if cleanup_profile == "default":
            return self._clean_default(markdown_text)
        if cleanup_profile == "vi_scientific_paper":
            return self._clean_vi_scientific_paper(markdown_text)
        raise ValueError("cleanup_profile must be one of: none, default, vi_scientific_paper")

    def convert_to_markdown(
        self,
        input_path: str,
        output_path: str,
        cleanup_profile: str = "default",
        engine: str = "markitdown",
    ) -> str:
        input_file = Path(input_path)
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if engine == "markitdown":
            markdown_text = self._extract_markdown(input_file)
        else:
            raise ValueError("engine must be one of: markitdown")
        markdown_text = self._apply_cleanup(markdown_text, cleanup_profile)
        if not markdown_text.endswith("\n"):
            markdown_text += "\n"

        output_file.write_text(markdown_text, encoding="utf-8")
        return str(output_file)
