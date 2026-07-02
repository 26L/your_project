"""평가 하니스 — 같은 평가셋을 어떤 기법에든 돌려 지표를 집계한다 (CLAUDE.md §6).

기법을 전혀 모른 채 공통 RagBackend.query() 만 호출한다.
질문 유형(single/multi/relational)별 분해와 선택적 LLM-as-judge 채점을 지원한다.
"""
from __future__ import annotations

import time
from statistics import mean
from typing import Any, Sequence

from ..core.interface import RagBackend
from .dataset import EvalItem
from .judge import judge_answer
from .metrics import dedupe_preserve, keyword_recall, retrieval_metrics

_AGG_KEYS = ("hit", "recall", "precision", "mrr", "keyword_recall", "judge_correct", "latency_s")


def _avg(values: Sequence[Any]) -> float | None:
    nums = [v for v in values if v is not None]
    return round(mean(nums), 4) if nums else None


def _aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    agg = {key: _avg([r[key] for r in rows]) for key in _AGG_KEYS}
    agg["n"] = len(rows)
    return agg


def run_eval(
    backend: RagBackend,
    items: Sequence[EvalItem],
    k: int,
    judge_llm: Any | None = None,
) -> dict[str, Any]:
    per_item: list[dict[str, Any]] = []
    for it in items:
        t0 = time.perf_counter()
        res = backend.query(it.question)
        latency = round(time.perf_counter() - t0, 3)

        retrieved = dedupe_preserve([c.source for c in res.contexts])
        row = {
            "question": it.question,
            "type": it.type,
            "latency_s": latency,
            **retrieval_metrics(retrieved, it.relevant_sources, k),
            "keyword_recall": keyword_recall(res.answer, it.answer_keywords),
            "judge_correct": (
                judge_answer(judge_llm, it.question, it.reference_answer, res.answer)
                if judge_llm is not None
                else None
            ),
            "retrieved": retrieved[:k],
            "answer": res.answer,
        }
        per_item.append(row)

    # 유형별 분해
    types = sorted({r["type"] for r in per_item})
    by_type = {t: _aggregate([r for r in per_item if r["type"] == t]) for t in types}

    return {
        "aggregate": _aggregate(per_item),
        "by_type": by_type,
        "per_item": per_item,
    }
