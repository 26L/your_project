"""검색/답변 지표. 기법 무관 — QueryResult 의 contexts(출처)와 answer 만 사용한다."""
from __future__ import annotations

from typing import Sequence


def dedupe_preserve(sources: Sequence[str | None]) -> list[str]:
    """순서 유지하며 출처 중복 제거(None 제외). 같은 문서의 여러 청크는 1개로 본다."""
    seen: set[str] = set()
    out: list[str] = []
    for s in sources:
        if s is None or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def retrieval_metrics(
    retrieved_sources: Sequence[str | None],
    relevant_sources: Sequence[str],
    k: int,
) -> dict[str, float | None]:
    """문서(출처) 단위 검색 지표. relevant 가 비면 None(채점 생략)."""
    relevant = set(relevant_sources)
    if not relevant:
        return {"hit": None, "recall": None, "precision": None, "mrr": None}

    ranked = dedupe_preserve(retrieved_sources)[:k]
    found = set(ranked) & relevant

    mrr = 0.0
    for rank, s in enumerate(ranked, start=1):
        if s in relevant:
            mrr = 1.0 / rank
            break

    return {
        "hit": 1.0 if found else 0.0,
        "recall": len(found) / len(relevant),
        "precision": len(found) / len(ranked) if ranked else 0.0,
        "mrr": mrr,
    }


def keyword_recall(answer: str, keywords: Sequence[str]) -> float | None:
    """답변 품질의 거친 프록시 — 키워드 포함 비율. 키워드 없으면 None."""
    if not keywords:
        return None
    low = answer.lower()
    present = sum(1 for kw in keywords if kw.lower() in low)
    return present / len(keywords)
