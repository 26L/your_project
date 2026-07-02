"""Standard RAG — 벡터(의미) 검색 → top-k 청크 → LLM 생성 (베이스라인)."""
from __future__ import annotations

from typing import Any

from ._common import LlamaIndexBackend


class StandardRAG(LlamaIndexBackend):
    name = "standard"

    def _make_engine(self) -> Any:
        return self._index.as_query_engine(similarity_top_k=self.cfg.top_k)
