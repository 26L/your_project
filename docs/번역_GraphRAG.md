# [번역·정리] GraphRAG — From Local to Global: A GraphRAG Approach to Query-Focused Summarization

> **로컬에서 글로벌로: 질의집중 요약을 위한 GraphRAG 접근** · Edge et al., Microsoft Research · 2024 · [arXiv 2404.16130](https://arxiv.org/abs/2404.16130) · [github.com/microsoft/graphrag](https://github.com/microsoft/graphrag)
> **우리 프로젝트가 직접 구현·비교한 논문.** 벡터 RAG가 못 푸는 "전역(global) sensemaking"을 지식그래프+커뮤니티 요약으로 푼다. (학습용 번역 + ragbench 연결)

---

## 초록 (Abstract) — 전문 번역

외부 지식원에서 관련 정보를 검색하는 RAG는 LLM이 **사적/미공개 문서 모음**에 대해 답하게 해준다. 그러나 RAG는 **"이 데이터셋의 주요 주제는 무엇인가?"** 같은 **전체 코퍼스를 향한 전역(global) 질문에는 실패**한다. 이는 본질적으로 **검색 과제가 아니라 질의집중 요약(QFS, Query-Focused Summarization)** 과제이기 때문이다. 한편 기존 QFS 방법은 전형적 RAG가 색인하는 텍스트 양으로 **확장되지 않는다.**

두 상반된 방법의 장점을 결합하기 위해 **GraphRAG** — 사용자 질문의 일반성과 원문 양 **양쪽으로 확장되는**, 사적 텍스트 코퍼스 QA용 그래프 기반 접근 — 을 제안한다. LLM으로 그래프 인덱스를 **2단계**로 구축한다: 먼저 원문에서 **엔티티 지식그래프**를 도출하고, 그다음 밀접히 관련된 엔티티 그룹마다 **커뮤니티 요약**을 **미리 생성**한다. 질문이 오면 각 커뮤니티 요약으로 **부분 응답**을 만들고, 모든 부분 응답을 다시 요약해 **최종 응답**을 낸다. 100만 토큰 규모의 전역 sensemaking 질문에서, GraphRAG는 기존 RAG 베이스라인 대비 답변의 **포괄성(comprehensiveness)과 다양성(diversity)** 을 크게 향상시켰다.

---

## 1. 핵심 개념 — 벡터 RAG(로컬) vs sensemaking(글로벌)

- **벡터 RAG**: 질의와 **국소적으로(local)** 관련된 소수 레코드를 검색 → 답. "특정 사실"엔 좋음.
- **한계**: **sensemaking**(전체 데이터셋을 아우르는 이해 — "지난 10년간 학제간 연구가 과학 발견에 어떤 영향을 줬나?") 질문엔 부적합. 이건 **전역 요약**이 필요.
- LLM(GPT·Llama·Gemini)은 sensemaking을 잘하지만, **데이터가 커서 RAG가 필요할 때** 벡터 RAG는 전역 요약을 지원 못 함.
- **GraphRAG**: 큰 코퍼스 전체에 대해 sensemaking을 가능케 하는 그래프 기반 RAG.

---

## 2. 방법 (Methods) — 파이프라인 (Figure 1)

```
[색인 시간 Indexing Time]              [질의 시간 Query Time]
원문 Source Documents                  최종 답 Global Answer
   │ 청킹                                 ↑ 질의집중 요약
텍스트 청크 Text Chunks                 커뮤니티 답변 Community Answers
   │ LLM 추출                              ↑ 질의집중 요약
엔티티·관계 Entities & Relationships    커뮤니티 요약 Community Summaries
   │ 요약 집약                             ↑ 요약
지식그래프 Knowledge Graph ── 커뮤니티 탐지 ──► 그래프 커뮤니티 Graph Communities
```

### 2.1 원문 → 텍스트 청크
- 코퍼스를 청크로 분할. **청크 크기 = 근본 설계 결정**: 크면 LLM 호출↓(비용↓)이나 **청크 앞부분 정보의 recall 저하**. (*우리 프로젝트가 겪은 청크 트레이드오프와 동일*)

### 2.2 텍스트 청크 → 엔티티·관계
- LLM이 청크에서 **엔티티 + 관계 + 짧은 설명(description)** 추출. **claims**(엔티티에 관한 사실 진술 — 날짜·사건 등)도 추출 가능.
- 예: "NeoChip은 Quantum Systems가 2016년 인수했다" → 엔티티 `NeoChip`(설명), `Quantum Systems`(설명), 관계(설명).
- 도메인 few-shot 예시로 프롬프트를 특화 가능(과학·의료·법률).
- → *이게 정확히 우리 `graphrag_e2b`의 "이름|유형|설명" 추출. 우리는 이 description 채움이 핵심이라 규명함(§10.1).*

### 2.3 엔티티·관계 → 지식그래프
- 추출은 **추상적 요약(abstractive summarization)** — 여러 문서에서 같은 요소가 반복 추출됨.
- 중복 인스턴스를 **노드/엣지로 집약**: 엔티티 설명 집약, **관계 인스턴스 수 = 엣지 가중치**. claims도 집약.
- **엔티티 매칭(중복 이름 통합)**: 본 논문은 exact string matching. soft matching도 가능. **그래프는 중복에 비교적 강함**(이후 요약 단계에서 클러스터링되므로). (*우리 L2 정규화·병합과 연결*)

### 2.4 지식그래프 → 그래프 커뮤니티
- **Leiden 커뮤니티 탐지**(계층적) — 강하게 연결된 노드 그룹으로 분할. 하위 커뮤니티를 재귀 탐지해 리프까지.
- 그래프의 **모듈성(modularity)** 을 활용 — 밀접 노드들을 커뮤니티로. (*우리 L5 = Neo4j GDS Louvain*)

### 2.5 그래프 커뮤니티 → 커뮤니티 요약
- 커뮤니티마다 **bottom-up LLM 요약** 생성. 상위 계층 요약은 하위 요약을 재귀 포함. → **코퍼스 전역 설명·통찰** 제공.

### 2.6 질의 (map-reduce 전역 검색)
- **map**: 각 커뮤니티 요약이 질의에 대한 **부분 답변**을 병렬 생성.
- **reduce**: 부분 답변들을 결합해 **최종 전역 답변**. → 모든 커뮤니티 요약에 대한 질의집중 요약.

---

## 3. 평가 — 적응형 벤치마킹 + LLM-judge

- **적응형 벤치마킹(adaptive benchmarking)**: HotpotQA·MultiHop-RAG 등 기존 벤치는 **명시적 사실검색(벡터 RAG)** 지향 → 전역 sensemaking 평가엔 부적합. 그래서 **페르소나 생성으로 전역 sensemaking 질문 셋을 LLM이 생성**(정답 없음). *(우리가 자체 평가셋 만든 것과 같은 문제의식 — CLAUDE.md §10.6)*
- **정답이 없으므로 LLM-as-judge 비교평가**: 두 시스템 답을 **포괄성·다양성** 기준으로 비교. "claims" 통계로도 검증.
- **결과**: **GPT-4 사용 시 GraphRAG가 벡터 RAG를 크게 능가**(전역 sensemaking에서).

---

## 💡 학습 포인트 · 우리 프로젝트(ragbench) 연결

| GraphRAG 논문 | ragbench에서 |
| --- | --- |
| 엔티티+관계+설명 추출 | `graphrag_e2b`의 산문추출("이름\|유형\|설명") |
| 중복 엔티티 집약(엔티티 매칭) | 우리 L2 정규화/병합(`normalize_name`) |
| Leiden 커뮤니티 탐지 | 우리 L5 = Neo4j GDS Louvain |
| 커뮤니티 요약(bottom-up) | 우리 `build_community_summaries.py` |
| map-reduce 전역 답변 | 우리는 단순화(요약을 검색에 주입, `_l5`) — **정통 GraphRAG는 map-reduce** |
| 벡터 RAG는 global 못 함 | 우리 실측: graph+L5가 global judge에서 평면검색 앞섰던 지점 |
| 전역 평가셋을 LLM 생성 | 우리 자체 가상 평가셋(§10.6 도메인 문제와 직결) |

**우리 결론과의 관계**: 논문은 "GraphRAG > 벡터 RAG(전역)"를 GPT-4로 보임. **우리는 여기서 한 발 더 나가** — ① 약judge→강judge 교차검증으로 그래프 점수 부풀림을 잡고, ② **검색층이 그래프 단일이면 핀포인트에 약함**을 규명, ③ **그래프+직접벡터 하이브리드**로 평면 정밀도까지 회복(0.333→0.806). 즉 **"GraphRAG의 전역 강점 + 벡터 RAG의 국소 정밀도"를 검색층 융합으로 합친** 셈.

> ⚠️ **정통 GraphRAG와 우리 구현 차이**: 논문은 **map-reduce 전역검색 + 계층 커뮤니티 요약**을 씀. 우리 구현은 이를 단순화(요약 주입). 완전 재현하려면 microsoft/graphrag 채택 필요(CLAUDE.md §10.2).

> 다음: CausalRAG(2503.19878) — 검색에 **인과(causal) 관계**를 반영.
