"""Hybrid RAG — 벡터(의미) + BM25(키워드) 검색을 RRF로 융합 → LLM 생성."""
from __future__ import annotations

from typing import Any

from ._common import LlamaIndexBackend, ko_tokenize


class HybridRAG(LlamaIndexBackend):
    name = "hybrid"

    def _make_engine(self) -> Any:
        from llama_index.core.query_engine import RetrieverQueryEngine
        from llama_index.core.retrievers import QueryFusionRetriever
        from llama_index.retrievers.bm25 import BM25Retriever

        vector = self._index.as_retriever(similarity_top_k=self.cfg.top_k)
        keyword = BM25Retriever.from_defaults(
            nodes=self._nodes(),
            similarity_top_k=self.cfg.top_k,
            tokenizer=ko_tokenize,
        )
        fusion = QueryFusionRetriever(
            [vector, keyword],
            similarity_top_k=self.cfg.top_k,
            num_queries=1,            # LLM 질의 확장 끔(비용/지연 절감)
            mode="reciprocal_rerank",  # RRF
            use_async=False,
            llm=self.llm,
        )
        return RetrieverQueryEngine.from_args(fusion, llm=self.llm)
