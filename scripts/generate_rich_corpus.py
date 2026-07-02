"""뼈대 생성 문서를 LLM(Gemini)으로 '내용 있는 본문'으로 다시 작성한다.

기존 generate_large_corpus.py 의 구조 데이터(직원·프로젝트)를 재사용하되, 각 문서의
본문을 LLM이 구체적 서술·고유 사실·자연스러운 산문으로 작성한다. 문서 간 인용은
문서명(「...」)으로. thinking 비활성(비용·MAX_TOKENS 방지).

사용:
  .venv/bin/python scripts/generate_rich_corpus.py 5     # 5건만(샘플, 비용 측정)
  .venv/bin/python scripts/generate_rich_corpus.py       # 전체
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from generate_large_corpus import EMP, MANAGER, MONTHS, PROJECTS, write  # noqa: E402

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
MODEL = "gemini-2.5-flash"
_CFG = types.GenerateContentConfig(
    temperature=0.7,
    max_output_tokens=2200,
    thinking_config=types.ThinkingConfig(thinking_budget=0),  # 생성엔 thinking 불필요
)

USAGE = {"in": 0, "out": 0, "n": 0}

SYS = (
    "당신은 가상 기업 '주식회사 하울'의 문서 작성 담당자다. 아래 사실을 바탕으로 실제 사내 문서처럼 "
    "구체적이고 자연스러운 한국어 문서를 작성하라. 규칙: (1) 제목은 '# '로 시작. (2) 머리말에 분류·부서·날짜 등 메타를 불릿으로. "
    "(3) 본문은 2~4개 섹션, 각 섹션은 여러 문장의 산문으로 구체적 내용을 채운다(항목 나열만 하지 말 것). "
    "(4) 관련 문서는 「문서명」 형태로 자연스럽게 인용. (5) 주어진 사실과 모순 금지, 없는 수치는 그럴듯하게 보강. "
    "(6) 마크다운만 출력(설명·코드펜스 금지)."
)


def gen(prompt: str) -> str:
    r = _client.models.generate_content(model=MODEL, contents=f"{SYS}\n\n[작성 의뢰]\n{prompt}", config=_CFG)
    u = r.usage_metadata
    USAGE["in"] += u.prompt_token_count or 0
    USAGE["out"] += (u.candidates_token_count or 0) + (getattr(u, "thoughts_token_count", 0) or 0)
    USAGE["n"] += 1
    return (r.text or "").strip()


def projects_of(name):
    return [p for p in PROJECTS if name == p[2] or name in p[3]]


# 생성 작업 목록(함수, 상대경로) 빌드 — 샘플 제한을 위해 리스트로
def build_tasks():
    tasks = []
    # 직원 프로필
    for name, dept, title, joined in EMP:
        mgr, mtitle = MANAGER[dept]
        projs = projects_of(name)
        facts = (f"직원 프로필. 이름 {name}, 부서 {dept}, 직급 {title}, 입사일 {joined}. "
                 f"상사 {mgr} {mtitle}. 참여 프로젝트: {', '.join(p[0] for p in projs) or '없음(부서 공통 업무)'}. "
                 f"연차·복지·교육은 「연차 유급휴가 규정」·「복지 규정」·「직원 직무교육 안내」를 따름. "
                 f"담당 업무·강점·최근 기여를 구체적으로 서술.")
        tasks.append((f"08_직원프로필/{name}_프로필.md", facts))
    # 프로젝트 계획서 + 진행보고
    for name, div, lead, members, equip, goal in PROJECTS:
        facts = (f"프로젝트 계획서. 프로젝트명 {name}, 주관부서 {div}, 책임자 {lead}, 참여자 {', '.join(members)}, "
                 f"사용 장비 {', '.join(equip)}(「장비 장부」 등록), 목표: {goal}. 예산은 「경비 처리 규정」, 보고는 「보고서 양식 가이드」. "
                 f"배경·목표·추진전략·일정(2025 상반기)·기대효과를 구체적으로.")
        tasks.append((f"09_프로젝트/{name}_계획서.md", facts))
        for mi, m in enumerate(MONTHS):
            who = members[mi % len(members)]
            facts = (f"프로젝트 {m}월 진행보고. 프로젝트 {name}({div}), 책임자 {lead}, 이번 달 주担당 {who}, 장비 {', '.join(equip)}. "
                     f"{m}월 실제 진행 내용(구체적 작업·수치·협업), 발생 이슈와 해결, 다음 달 계획을 서술. 비용은 「경비 처리 규정」 정산.")
            tasks.append((f"09_프로젝트/{name}_{m}월_진행보고.md", facts))
    # 월간 업무보고
    DEPTS = ["농업사업부", "인력사업부", "신사업개발부", "인사팀", "경영지원팀"]
    for dept in DEPTS:
        dprojs = [p for p in PROJECTS if p[1] == dept]
        mgr, mtitle = MANAGER[dept]
        for m in MONTHS:
            facts = (f"{dept} {m}월 월간업무보고. 작성자 {mgr} {mtitle}. 부서 프로젝트: {', '.join(p[0] for p in dprojs) or '지원 업무'}. "
                     f"{m}월 주요 실적·예산 집행(「경비 처리 규정」)·인력 현황(「직원 현황」·「인력 통계」)·다음 달 계획을 구체적으로.")
            tasks.append((f"10_월간보고/{dept}_{m}월.md", facts))
    # 회의록
    for m in MONTHS:
        att = ", ".join(f"{MANAGER[d][0]} {MANAGER[d][1]}" for d in DEPTS)
        facts = (f"{m}월 월례 전체회의 회의록. 참석 {att}. 각 사업부 {m}월 현황 공유, 부서 협업 안건, 예산(「경비 처리 규정」)·일정(「부서 일정」)·채용(「채용 규정」) 관련 결정사항과 후속조치를 구체적으로.")
        tasks.append((f"11_회의록/전체회의_{m}월.md", facts))
    return tasks


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    tasks = build_tasks()
    if limit:
        tasks = tasks[:limit]
    print(f"생성 대상: {len(tasks)}건 (모델 {MODEL}, thinking off)")
    for i, (rel, facts) in enumerate(tasks, 1):
        body = gen(facts)
        if body:
            write(rel, body)
        if i % 10 == 0 or i == len(tasks):
            print(f"  {i}/{len(tasks)} 완료  (누적 in={USAGE['in']:,} out={USAGE['out']:,} 토큰)")
    # 비용 추정(대략): flash 가격은 변동 가능 — 토큰 기준 참고치
    in_per_m, out_per_m = 0.30, 2.50  # USD per 1M tokens (대략)
    cost = USAGE["in"] / 1e6 * in_per_m + USAGE["out"] / 1e6 * out_per_m
    print(f"\n토큰 합계: in={USAGE['in']:,}  out={USAGE['out']:,}  (문서 {USAGE['n']}건)")
    print(f"대략 비용: ${cost:.4f}  (1건당 ${cost/max(USAGE['n'],1):.4f}) — 단가는 변동 가능, 참고치")
    if limit:
        full = len(build_tasks())
        print(f"전체 {full}건 추정: ${cost/max(USAGE['n'],1)*full:.2f}")


if __name__ == "__main__":
    main()
