"""RAG 기법 공통 베이스 — LlamaIndex VectorStoreIndex 기반 인덱싱/로딩 공유.

검색 방식만 다른 백엔드(standard/bm25/hybrid)는 `_make_engine()` 만 구현한다.
"""
from __future__ import annotations

import os
import re
from typing import Any, Sequence

from ..core.config import Config
from ..core.interface import QueryResult, RagBackend, RetrievedContext

_TOKEN = re.compile(r"[A-Za-z0-9가-힣]+")


def ko_tokenize(text: str) -> list[str]:
    """한국어/영문/숫자 토큰화(어절·연속문자 단위). 형태소 분석 없이 BM25용 1차 토크나이저."""
    return _TOKEN.findall(text.lower())


class LlamaIndexBackend(RagBackend):
    """VectorStoreIndex 로 인덱싱하고, 검색기만 바꿔 끼우는 공통 베이스."""

    name = "base"

    def __init__(self, cfg: Config, llm: Any, embed_model: Any):
        self.cfg = cfg
        self.llm = llm
        self.embed_model = embed_model
        self.persist_dir = os.path.join(cfg.storage_dir, self.name)
        self._index = None

        from llama_index.core import Settings
        from llama_index.core.node_parser import SentenceSplitter

        Settings.llm = llm
        Settings.embed_model = embed_model
        Settings.node_parser = SentenceSplitter(
            chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap
        )

    def index(self, documents: Sequence[Any]) -> None:
        from llama_index.core import VectorStoreIndex

        self._index = VectorStoreIndex.from_documents(list(documents))
        self._index.storage_context.persist(persist_dir=self.persist_dir)

    def _ensure_loaded(self) -> None:
        if self._index is not None:
            return
        if not os.path.isdir(self.persist_dir):
            raise FileNotFoundError(
                f"인덱스가 없습니다: {self.persist_dir}. 먼저 'ragbench index --method {self.name}' 를 실행하세요."
            )
        from llama_index.core import StorageContext, load_index_from_storage

        sc = StorageContext.from_defaults(persist_dir=self.persist_dir)
        self._index = load_index_from_storage(sc)

    def _nodes(self) -> list[Any]:
        return list(self._index.docstore.docs.values())

    def _make_engine(self) -> Any:
        raise NotImplementedError

    def query(self, question: str) -> QueryResult:
        self._ensure_loaded()
        resp = self._make_engine().query(question)
        contexts = [
            RetrievedContext(
                text=node.node.get_content(),
                source=node.node.metadata.get("file_name"),
                score=node.score,
                metadata=node.node.metadata,
            )
            for node in resp.source_nodes
        ]
        return QueryResult(
            answer=str(resp),
            contexts=contexts,
            metadata={"method": self.name, "top_k": self.cfg.top_k},
        )
