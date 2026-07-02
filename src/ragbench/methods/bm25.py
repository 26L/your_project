"""BM25 RAG — 키워드(렉시컬) 검색 → top-k 청크 → LLM 생성."""
from __future__ import annotations

from typing import Any

from ._common import LlamaIndexBackend, ko_tokenize


class BM25RAG(LlamaIndexBackend):
    name = "bm25"

    def _make_engine(self) -> Any:
        from llama_index.core.query_engine import RetrieverQueryEngine
        from llama_index.retrievers.bm25 import BM25Retriever

        retriever = BM25Retriever.from_defaults(
            nodes=self._nodes(),
            similarity_top_k=self.cfg.top_k,
            tokenizer=ko_tokenize,
        )
        return RetrieverQueryEngine.from_args(retriever, llm=self.llm)
