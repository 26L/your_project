# 아키텍처 · 코드 지도 (ARCHITECTURE)

> `ragbench` 코드베이스 탐색·수정용 가이드. **"무슨 코드가 어디 있고, X를 고치려면 어디를 보는지"** 에 초점. 설계 배경·실험 결과는 [CLAUDE.md](../CLAUDE.md), 진행 타임라인은 [LOG_WORK.md](LOG_WORK.md) 참고.

## 1. 한눈에 — 플러그형 설계

모든 RAG 기법을 **공통 인터페이스**(`RagBackend`) 뒤에 두고, 임베딩·LLM을 교체 축으로 둔다. 평가 하니스는 기법을 몰라도 된다.

```text
  CLI (cli.py)  ──build_llm/build_embed_model──►  LLM·임베딩 백엔드
       │                                          (llms/, embeddings/)
       │ build_backend(name, cfg, llm, embed)
       ▼
  registry.py  ──►  RagBackend 구현 (methods/)
       │                 · index(documents) -> 영속 인덱스
       │                 · query(question)  -> QueryResult{answer, contexts, metadata}
       ▼
  eval/harness.py  ──►  지표 집계 (metrics·judge)  ──►  results/*.json
```

- **핵심 계약**: [core/interface.py](../src/ragbench/core/interface.py) — `RagBackend`(ABC), `QueryResult`, `RetrievedContext`
- **기법 등록**: [registry.py](../src/ragbench/registry.py) — 이름→클래스 매핑(`METHODS` dict). **새 기법은 여기 한 줄 + methods/ 어댑터**.

## 2. 요청 흐름 (3가지 명령)

CLI 진입점: [cli.py](../src/ragbench/cli.py) `main()` → 서브커맨드 `index` / `query` / `eval` / `compare`. 콘솔스크립트 `ragbench`([pyproject.toml](../pyproject.toml)).

| 명령 | 흐름 | 주요 파일 |
| --- | --- | --- |
| **index** | 문서 로드 → 기법.index() → 저장 | `cmd_index` → [ingest/loader.py](../src/ragbench/ingest/loader.py) → `methods/*` |
| **query** | 기법.query() → 답변+출처 | `cmd_query` → `methods/*._make_engine()` |
| **eval** | 평가셋 로드 → 각 문항 query → 지표 집계 | `cmd_eval` → [eval/harness.py](../src/ragbench/eval/harness.py) |

**공통 백엔드 골격**: [methods/_common.py](../src/ragbench/methods/_common.py) `LlamaIndexBackend` — `persist_dir`, `index()`/`query()` 기본 구현, `ko_tokenize`(한국어 BM25 토크나이저). 대부분의 기법이 이걸 상속.

## 3. 디렉토리 지도

```text
src/ragbench/
  core/
    interface.py   ★ RagBackend(ABC)·QueryResult·RetrievedContext — 모든 기법의 계약
    config.py      ★ Config·LLMConfig·EmbedConfig (YAML 로딩, Config.load)
  ingest/loader.py   문서 로드 (SimpleDirectoryReader 래핑)
  embeddings/factory.py  build_embed_model — google / openai / local(e5) 분기
  llms/factory.py        build_llm — google(Gemini) / anthropic / ollama 분기
  methods/
    _common.py     LlamaIndexBackend(공통 골격)·ko_tokenize
    standard.py    StandardRAG (의미 검색, VectorStoreIndex)
    bm25.py        BM25RAG (키워드)
    hybrid.py      HybridRAG (벡터+BM25 RRF)
    graphrag.py    GraphRAG·GraphRAGSchema·GraphRAGDynamic (추출기 3종 비교)
    graphrag_e2b.py  ★ E2B 계열 (아래 §5) — 이 프로젝트 핵심
  eval/
    dataset.py     EvalItem·load_eval_set (평가셋 YAML 파싱)
    harness.py     run_eval — 문항별 query + 지표 집계 + per_item 저장
    metrics.py     retrieval_metrics(recall@k·precision·mrr)·keyword_recall
    judge.py       judge_answer (LLM-as-judge 0/1)
  registry.py      ★ METHODS dict (이름→기법)·build_backend
  cli.py           index/query/eval/compare 진입점
config/            *.yaml (기법·임베딩·모델·청킹·top-k·평가셋)
scripts/           코퍼스 생성·Neo4j 적재·커뮤니티 요약
data/·storage/·results/  코퍼스·인덱스·결과 (git 제외)
```

## 4. 핵심 계약 — RagBackend

[core/interface.py](../src/ragbench/core/interface.py):

```python
class RagBackend(ABC):
    name: str
    def index(self, documents) -> None: ...      # 구축·저장
    def query(self, question) -> QueryResult: ...  # {answer, contexts, metadata}
```

- `QueryResult.contexts`: `RetrievedContext[]` (출처 청크/노드) — 평가의 recall@k 계산에 쓰임.
- 새 기법은 이 계약만 지키면 평가 하니스·CLI가 그대로 동작.

## 5. graphrag_e2b 계열 (가장 복잡 — 상세)

[methods/graphrag_e2b.py](../src/ragbench/methods/graphrag_e2b.py). **같은 그래프를 쓰고 "검색 방식"만 다른** 4개 기법 + 추출·정규화 유틸.

### 5.1 추출 (그래프 구축)
- `parse_extraction()` — LLM 산문("이름|유형|설명") → (엔티티, 관계) 파싱
- `normalize_name/type/rel()` — 정규화(「」 병합·타입 교정·관계 표면형 통합)
- `_build_extractor_cls()` → `E2BExtractor` — 청크마다 LLM 호출해 EntityNode+Relation 생성. `index()`는 부모 `GraphRAG.index()`(PropertyGraphIndex) 사용.

### 5.2 검색 (기법별로 다름 — `_make_engine`)
| 클래스 | 검색 방식 | 비고 |
| --- | --- | --- |
| `GraphRAGE2B` | 그래프 검색 + `E5Rerank` 재순위 | 기준 |
| `GraphRAGE2BL5` | 위 + 커뮤니티 요약 주입 | `_load_summaries()`(community_summaries.json), persist_dir=`storage/graphrag_e2b` 재사용 |
| `GraphRAGE2BAdaptive` | 위 + 질의 유형 라우팅 | `_GraphCommunityRetriever._inject_summaries`(router LLM) |
| **`GraphRAGE2BHybrid`** | **그래프 + 직접 청크벡터 RRF 융합** | `standard` 인덱스 재사용, `QueryFusionRetriever` |

### 5.3 검색 보조 (팩토리로 생성 — import 순서 회피)
- `_build_reranker_cls()` → `E5Rerank` — 질의-노드 e5 유사도 재순위. **`clean()`**: PropertyGraphIndex가 붙이는 트리플 접두사 제거 후 임베딩(e5 512토큰 오염 버그 수정 — CLAUDE.md §10.5).
- `_build_community_retriever_cls()` → `_GraphCommunityRetriever` — 그래프검색+재순위+커뮤니티요약 결합, 라우터로 주입 판정.

## 6. "이걸 고치려면 여기를 보라" (작업→파일 지도)

| 하고 싶은 것 | 볼 곳 |
| --- | --- |
| **새 RAG 기법 추가** | `methods/` 에 `RagBackend`(또는 `LlamaIndexBackend`) 어댑터 + [registry.py](../src/ragbench/registry.py) `METHODS` 한 줄 |
| **임베딩 백엔드 추가/교체** | [embeddings/factory.py](../src/ragbench/embeddings/factory.py) `build_embed_model` 분기 + config `embed.provider/model` |
| **생성 LLM 추가/교체** | [llms/factory.py](../src/ragbench/llms/factory.py) `build_llm` 분기 (google/anthropic/ollama) |
| **Gemini thinking 끄기(비용)** | config `llm.thinking: false` → factory google 분기 |
| **청킹·top-k·모델 변경** | `config/*.yaml` (하드코딩 금지 — CLAUDE.md §7) |
| **그래프 추출 품질(엔티티·관계)** | graphrag_e2b.py `parse_extraction`·`normalize_*`·`E2BExtractor` |
| **그래프 검색/재순위 개선** | graphrag_e2b.py `_make_engine`·`E5Rerank`·`_GraphCommunityRetriever` |
| **평가 지표 추가/수정** | [eval/metrics.py](../src/ragbench/eval/metrics.py)·[eval/harness.py](../src/ragbench/eval/harness.py) |
| **LLM-judge 채점 로직** | [eval/judge.py](../src/ragbench/eval/judge.py) |
| **평가셋(문항·정답) 편집** | [config/eval_sample.yaml](../config/eval_sample.yaml) (`question·type·relevant_sources·answer_keywords·reference_answer`) |
| **CLI 명령/옵션** | [cli.py](../src/ragbench/cli.py) |

## 7. 설정 시스템

[core/config.py](../src/ragbench/core/config.py) — `Config.load(path)`가 YAML→dataclass. 필드: `llm`(provider/model/temperature/max_tokens/num_ctx/thinking), `embed`(provider/model), `chunk_size`·`chunk_overlap`·`top_k`·`data_dir`·`storage_dir`.

| config | 용도 |
| --- | --- |
| `default.yaml` | Gemini 생성+임베딩 (기본) |
| `ollama.yaml` | 로컬 — gemma(Ollama)+e5, chunk 1024, num_ctx 8192 |
| `local.yaml` | Gemini 생성 + 로컬 e5 |
| `gemini_extract.yaml` | Gemini 추출·생성·judge (thinking off) + 로컬 e5 |
| `eval_sample.yaml` / `eval_global_only.yaml` | 평가셋 |

## 8. 저장소·데이터 레이아웃 (git 제외)

- `data/company/` — 코퍼스(가상 회사 문서). `data/noise/` — 노이즈 토글.
- `storage/<method>/` — 기법별 영속 인덱스. graphrag 계열: `property_graph_store.json`(노드·관계)·`default__vector_store.json`(임베딩)·`docstore.json`(청크)·`community_summaries.json`(L5). `_gemma_v2`·`_gemini_sub` 등은 비교용 백업.
- `results/` · `results/local_eval/` — 평가 결과 JSON(`per_item` 포함).

## 9. 외부 인프라

| 서비스 | 용도 | 설정 |
| --- | --- | --- |
| **Ollama** (M2 등) | 로컬 LLM(gemma) | `OLLAMA_BASE_URL` 환경변수 |
| **Neo4j + GDS** | 그래프 시각화·커뮤니티(Louvain)·PageRank | [docker-compose.yml](../docker-compose.yml) (`docker compose up -d`) |
| **Google AI Studio** | Gemini 생성·임베딩·judge | `GEMINI_API_KEY` ([.env](../.env.example)) |

관련 스크립트: [scripts/migrate_graph_to_neo4j.py](../scripts/migrate_graph_to_neo4j.py)(그래프→Neo4j 적재), [scripts/build_community_summaries.py](../scripts/build_community_summaries.py)(GDS Louvain+LLM 요약), [scripts/generate_rich_corpus.py](../scripts/generate_rich_corpus.py)(코퍼스 생성).
