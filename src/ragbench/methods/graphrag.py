"""GraphRAG 계열 — LLM로 엔티티·관계를 추출해 속성 그래프를 만들고 그래프 검색으로 답변.

추출기만 다른 3종을 비교 대상으로 둔다(같은 인터페이스):
- graphrag         : SimpleLLMPathExtractor — 평면 트리플(타입·속성 없음). 기준선.
- graphrag_schema  : SchemaLLMPathExtractor — 도메인 스키마 강제(closed-world) + 속성(description).
- graphrag_dynamic : DynamicLLMPathExtractor — 시드 온톨로지(open-world) + 속성(description).

주의: 인덱싱 시 청크마다 LLM 추출 → 시간 소요. 임베딩은 로컬 권장(rate limit 회피).
"""
from __future__ import annotations

import dataclasses
import os
from typing import Any, Sequence

from ._common import LlamaIndexBackend
from ..llms.factory import build_llm

# 도메인 온톨로지 (하울 코퍼스) — Schema/Dynamic 공통 시드
ENTITY_TYPES = ["직원", "부서", "프로젝트", "규정", "양식", "장비", "회의", "법령", "직급"]
RELATION_TYPES = ["소속", "참여", "사용", "담당", "보고", "인용", "근거", "승인"]


class GraphRAG(LlamaIndexBackend):
    """기준선 — SimpleLLMPathExtractor(평면 트리플)."""

    name = "graphrag"
    _big = None

    def _big_llm(self) -> Any:
        # 추출·합성 모두 출력이 길어 max_tokens 여유 필요(2.5 thinking 토큰).
        if self._big is None:
            big = dataclasses.replace(
                self.cfg, llm=dataclasses.replace(self.cfg.llm, max_tokens=8192)
            )
            self._big = build_llm(big)
        return self._big

    def _extractor(self) -> Any:
        from llama_index.core.indices.property_graph import SimpleLLMPathExtractor

        return SimpleLLMPathExtractor(
            llm=self._big_llm(), max_paths_per_chunk=8, num_workers=2
        )

    def index(self, documents: Sequence[Any]) -> None:
        from llama_index.core import PropertyGraphIndex

        self._index = PropertyGraphIndex.from_documents(
            list(documents),
            llm=self.llm,
            embed_model=self.embed_model,
            kg_extractors=[self._extractor()],
            embed_kg_nodes=True,
            show_progress=True,
        )
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

    def _make_engine(self) -> Any:
        import nest_asyncio  # 그래프 검색기의 중첩 async 허용

        nest_asyncio.apply()
        return self._index.as_query_engine(
            similarity_top_k=self.cfg.top_k, llm=self._big_llm()
        )


class GraphRAGSchema(GraphRAG):
    """도메인 스키마 강제 추출(closed-world) + 엔티티/관계 description."""

    name = "graphrag_schema"

    def _extractor(self) -> Any:
        from typing import Literal

        from llama_index.core.indices.property_graph import SchemaLLMPathExtractor

        ent = Literal[tuple(ENTITY_TYPES)]
        rel = Literal[tuple(RELATION_TYPES)]
        validation = {e: RELATION_TYPES for e in ENTITY_TYPES}  # 우선 전조합 허용
        # 주의: props(description) 지정 시 JSON 스키마에 additionalProperties 발생 →
        # Gemini Developer API가 거부(Vertex만 지원) → props 없이 type 제약만 사용.
        return SchemaLLMPathExtractor(
            llm=self._big_llm(),
            possible_entities=ent,
            possible_relations=rel,
            kg_validation_schema=validation,
            strict=False,
            max_triplets_per_chunk=8,
            num_workers=2,
        )


class GraphRAGDynamic(GraphRAG):
    """시드 온톨로지 기반 동적 추출(open-world) + 엔티티/관계 description."""

    name = "graphrag_dynamic"

    def _extractor(self) -> Any:
        from llama_index.core.indices.property_graph import DynamicLLMPathExtractor

        return DynamicLLMPathExtractor(
            llm=self._big_llm(),
            allowed_entity_types=ENTITY_TYPES,
            allowed_relation_types=RELATION_TYPES,
            allowed_entity_props=["description"],
            allowed_relation_props=["description"],
            max_triplets_per_chunk=8,
            num_workers=2,
        )
