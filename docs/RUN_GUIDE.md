# 실행 가이드 (RUN GUIDE)

> ragbench 설치·실행·설정별 사용·트러블슈팅. 명령은 실제 검증된 것.

## 0. 사전 요구사항
| 항목 | 필요 | 비고 |
| --- | --- | --- |
| Python 3.10 + **uv** | 필수 | 패키지·가상환경 |
| GPU(CUDA) | 선택 | 로컬 임베딩(e5) 가속. 없으면 CPU |
| **Gemini API 키** | 클라우드 쓸 때 | `GEMINI_API_KEY` (ai.google.dev) |
| **Ollama** | 로컬 LLM 쓸 때 | gemma 등. `OLLAMA_BASE_URL` |
| **Docker** | 그래프 계열 쓸 때 | Neo4j 실행 |

## 1. 설치
```bash
uv venv --python 3.10 .venv
uv pip install -e .
```

## 2. 키 · 설정
```bash
cp .env.example .env      # GEMINI_API_KEY 등 채우기
```
`.env` 항목:
- `GEMINI_API_KEY=...` — Gemini 생성·임베딩·judge
- `OLLAMA_BASE_URL=http://<M2-IP>:11434` — 로컬 LLM(다른 머신이면 IP)
- `NEO4J_URL/USER/PASSWORD` — Neo4j(그래프 계열)

**설정 파일**(`config/*.yaml`) — 실행 시 `--config`로 선택:
| config | 생성 LLM | 임베딩 | 용도 |
| --- | --- | --- | --- |
| `default.yaml` | Gemini | Gemini | 기본(클라우드) |
| `ollama.yaml` | gemma(Ollama) | 로컬 e5 | **전부 로컬·무료** |
| `local.yaml` | Gemini | 로컬 e5 | 생성만 클라우드 |
| `gemini_extract.yaml` | Gemini(thinking off) | 로컬 e5 | 추출·생성·judge Gemini |

## 3. 기본 실행 3종 (index → query → eval)
```bash
# 인덱싱 (색인 구축·저장)
.venv/bin/ragbench index --method hybrid --config config/ollama.yaml --data data/company

# 질의 (답변+출처)
.venv/bin/ragbench query --method hybrid --config config/ollama.yaml "연차는 어떤 규정에 근거하나?"

# 평가 (지표 집계 → results/<method>.json)
.venv/bin/ragbench eval  --method hybrid --config config/ollama.yaml \
    --eval-set config/eval_sample.yaml --judge
```
- `--method`: `standard` `bm25` `hybrid` `graphrag` `graphrag_e2b` `graphrag_e2b_l5` `graphrag_e2b_adaptive` `graphrag_e2b_hybrid`
- `--judge`: LLM-as-judge 정답 채점(config의 llm 사용)
- `--data`: 코퍼스 폴더 (기본 `data/company`)

## 4. 설정별 실행

### (A) 전부 로컬·무료 (gemma + e5)
```bash
export OLLAMA_BASE_URL=http://<M2-IP>:11434
.venv/bin/ragbench index --method standard --config config/ollama.yaml --data data/company
.venv/bin/ragbench eval  --method standard --config config/ollama.yaml --eval-set config/eval_sample.yaml --judge
```

### (B) Gemini (추출·생성·judge) + 로컬 e5
```bash
# .env에 GEMINI_API_KEY 필요
.venv/bin/ragbench eval --method hybrid --config config/gemini_extract.yaml \
    --eval-set config/eval_sample.yaml --judge
```

## 5. 그래프 계열 실행 (graphrag_e2b*)
```bash
# 1) Neo4j 실행
docker compose up -d                 # http://localhost:7474

# 2) 그래프 추출·색인 (LLM 추출 → 시간 소요)
export OLLAMA_BASE_URL=http://<M2-IP>:11434
.venv/bin/ragbench index --method graphrag_e2b --config config/ollama.yaml --data data/company

# 3) Neo4j 적재 (시각화·GDS용)
.venv/bin/python scripts/migrate_graph_to_neo4j.py storage/graphrag_e2b

# 4) 커뮤니티 요약 (L5용)
.venv/bin/python scripts/build_community_summaries.py config/ollama.yaml storage/graphrag_e2b

# 5) 평가 (l5·adaptive·hybrid는 위 그래프 재사용)
.venv/bin/ragbench eval --method graphrag_e2b_hybrid --config config/ollama.yaml \
    --eval-set config/eval_sample.yaml --judge
```
> `graphrag_e2b_hybrid`는 **standard 인덱스도 필요**(직접 벡터검색용) — 먼저 `index --method standard` 실행.

## 6. 결과 비교
```bash
.venv/bin/ragbench compare               # results/*.json 라벨 비교표
```
- 결과 파일: `results/<method>.json`, `results/local_eval/*.json` (`per_item` 포함 → 재채점 가능)

## 7. 트러블슈팅 (실제 겪은 이슈)
| 증상 | 원인 | 해결 |
| --- | --- | --- |
| `MAX_TOKENS` 조기 종료 | gemini-2.5 thinking이 출력 토큰 소모 | config `max_tokens` ↑ (그래프 추출은 8192) |
| 입력이 4096에서 잘림 | Ollama 기본 `num_ctx=4096` | config `llm.num_ctx: 8192` |
| 임베딩 429(rate limit) | Gemini 임베딩 분당 한도 | **로컬 e5로 교체**(config embed: local) |
| 그래프 인덱싱 중첩 async 에러 | PropertyGraphIndex 검색기 | `nest_asyncio.apply()` (코드 내장됨) |
| Gemini 비용↑·느림 | thinking 켜짐 | config `llm.thinking: false` |
| 그래프 recall 낮음 | 재랭커 트리플 오염·단일 검색 | 재랭커 수정 + `graphrag_e2b_hybrid` 사용 |
| M2 gemma 매우 느림 | 온디바이스 8GB | 청크↑·문항↓, 또는 Gemini(클라우드) |

## 8. 빠른 시작 (한 번에)
```bash
uv venv --python 3.10 .venv && uv pip install -e .
cp .env.example .env                                  # 키 채우기
docker compose up -d                                  # (그래프 쓸 때)
.venv/bin/ragbench index --method hybrid --config config/ollama.yaml --data data/company
.venv/bin/ragbench eval  --method hybrid --config config/ollama.yaml --eval-set config/eval_sample.yaml --judge
```

> 자세한 코드 위치는 [ARCHITECTURE.md](ARCHITECTURE.md), 기술 문서는 [TECH_REFERENCE.md](TECH_REFERENCE.md).
