from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from app.core.constants import (
    RETRIEVAL_SCOPE_HYBRID_SYSTEM_AND_USER,
    SOURCE_TYPE_SYSTEM,
    SOURCE_TYPE_USER_UPLOAD,
)


@dataclass
class RetrievalBranch:
    name: str
    query: str
    metadata_filter: dict[str, Any]
    top_k: int

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalPlan:
    mode: str
    should_retrieve: bool
    branches: list[RetrievalBranch] = field(default_factory=list)
    reason: str | None = None

    def model_dump(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["branches"] = [branch.model_dump() for branch in self.branches]
        return payload


class RetrievalStrategy:
    def _filter_has_value(self, metadata_filter: dict[str, Any], key: str, value: Any) -> bool:
        if not isinstance(metadata_filter, dict):
            return False
        if metadata_filter.get(key) == value:
            return True
        for operator in ("$and", "$or"):
            nested_filters = metadata_filter.get(operator)
            if isinstance(nested_filters, list):
                return any(self._filter_has_value(item, key, value) for item in nested_filters)
        return False

    def _split_hybrid_filter(self, metadata_filter: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        filters = metadata_filter.get("$or") if isinstance(metadata_filter, dict) else None
        system_filter = {"source_type": SOURCE_TYPE_SYSTEM}
        user_filter = {"source_type": SOURCE_TYPE_USER_UPLOAD}
        if isinstance(filters, list):
            for item in filters:
                if self._filter_has_value(item, "source_type", SOURCE_TYPE_SYSTEM):
                    system_filter = item
                elif self._filter_has_value(item, "source_type", SOURCE_TYPE_USER_UPLOAD):
                    user_filter = item
        return system_filter, user_filter

    def plan(
        self,
        rewritten_question: str,
        intent_resolution: dict[str, Any],
        scope: str,
        metadata_filter: dict[str, Any],
    ) -> RetrievalPlan:
        if not intent_resolution.get("needs_retrieval", True):
            return RetrievalPlan(mode="none", should_retrieve=False, reason="Intent does not need retrieval.")

        intent = intent_resolution.get("intent", "ask_question")
        if scope == RETRIEVAL_SCOPE_HYBRID_SYSTEM_AND_USER or intent == "compare_documents":
            system_filter, user_filter = self._split_hybrid_filter(metadata_filter)
            return RetrievalPlan(
                mode="hybrid_compare",
                should_retrieve=True,
                branches=[
                    RetrievalBranch("system_chunks", rewritten_question, system_filter, 6),
                    RetrievalBranch("user_upload_chunks", rewritten_question, user_filter, 6),
                ],
                reason="Compare intent retrieves system and user-upload sources separately.",
            )

        if intent == "summarize_document":
            top_k = 12
            mode = "summarize"
        elif intent == "find_information":
            top_k = 8
            mode = "find_information"
        else:
            top_k = 5
            mode = "default"

        return RetrievalPlan(
            mode=mode,
            should_retrieve=True,
            branches=[RetrievalBranch("default", rewritten_question, metadata_filter, top_k)],
            reason="Single-source retrieval plan.",
        )
