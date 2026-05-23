import os
import subprocess
import sys
from pathlib import Path


def test_bge_embedding_real_system_markdown_ranks_fee_chunk_above_noise_in_subprocess():
    backend_root = Path(__file__).resolve().parents[1]
    code = r"""
import math
from pathlib import Path

from app.core.config import settings
from app.rag.embedding.bge_embedding import BGEEmbeddingService

settings.OPENAI_API_KEY = ""
doc_path = (
    Path("storage")
    / "markdown"
    / "system"
    / "system"
    / "sysdoc_b98ec8c6-aa4e-48cf-9d10-2a906ce92dd7"
    / "document.md"
)
text = doc_path.read_text(encoding="utf-8")

def find_line(*needles):
    normalized_needles = [needle.lower() for needle in needles]
    for line in text.splitlines():
        normalized_line = line.lower()
        if all(needle in normalized_line for needle in normalized_needles):
            return line.strip()
    raise AssertionError(f"Cannot find line containing: {needles}")

def cosine(left, right):
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0

query = "phi cong bo 100.000 dong va le phi 25.000 dong cua dang ky doanh nghiep tu nhan"
fee_line = find_line("100.000", "25.000")
deadline_line = find_line("03", "ng")
noise_line = find_line("b", "sao")

embedding_service = BGEEmbeddingService()
query_embedding, fee_embedding, deadline_embedding, noise_embedding = embedding_service.embed_texts(
    [query, fee_line, deadline_line, noise_line]
)
fee_score = cosine(query_embedding, fee_embedding)
deadline_score = cosine(query_embedding, deadline_embedding)
noise_score = cosine(query_embedding, noise_embedding)

print(f"fee_score={fee_score}")
print(f"deadline_score={deadline_score}")
print(f"noise_score={noise_score}")

assert len(query_embedding) > 0
assert fee_score > noise_score
assert max(fee_score, deadline_score) > noise_score
"""
    env = {
        **os.environ,
        "PYTHONPATH": ".",
        "OPENAI_API_KEY": "",
        "LANGSMITH_TRACING": "false",
        "LANGCHAIN_TRACING_V2": "false",
        "ANONYMIZED_TELEMETRY": "False",
        "POSTHOG_DISABLED": "true",
    }
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=backend_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=240,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "fee_score=" in result.stdout
    assert "Windows fatal exception" not in result.stderr, result.stdout + result.stderr
