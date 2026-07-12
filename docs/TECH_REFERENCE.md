# 기술 레퍼런스 — 공식 문서 · GitHub 링크

> ragbench가 쓰는 기술의 **공식 가이드·레포** 모음. 혼자 빌드/학습 시 1차 근거.
> ⚠️ URL·패키지 버전은 변할 수 있으니 접속 시 최신판 확인.

## 1. 프레임워크 핵심 — LlamaIndex
- 공식 문서: https://docs.llamaindex.ai
- GitHub: https://github.com/run-llama/llama_index
- 핵심 개념 페이지(문서에서 검색):
  - `VectorStoreIndex` — 벡터 색인·검색 (standard)
  - `PropertyGraphIndex` — 지식그래프 색인 (graphrag)
  - `QueryFusionRetriever` — 여러 retriever RRF 융합 (hybrid)
  - `BaseRetriever` / `BaseNodePostprocessor` — 커스텀 검색·재순위
  - `TransformComponent` — 커스텀 추출기(E2B)
  - `StorageContext`, `load_index_from_storage` — 영속화
  - `SimpleDirectoryReader`, `SentenceSplitter` — 로드·청킹

## 2. LLM · 임베딩 (LlamaIndex 통합 패키지)
| 용도 | 패키지 | 참고 |
| --- | --- | --- |
| Google Gemini 생성 | `llama-index-llms-google-genai` | https://docs.llamaindex.ai (검색: GoogleGenAI) |
| Google 임베딩 | `llama-index-embeddings-google-genai` | 〃 |
| Ollama(로컬) 생성 | `llama-index-llms-ollama` | 〃 |
| Anthropic 생성 | `llama-index-llms-anthropic` | 〃 |
| HuggingFace 임베딩(e5) | `llama-index-embeddings-huggingface` | 〃 |
| BM25 검색 | `llama-index-retrievers-bm25` | 〃 |
| Neo4j 그래프 저장 | `llama-index-graph-stores-neo4j` | 〃 |

**외부 서비스 문서**
- Google Gemini API: https://ai.google.dev/gemini-api/docs (모델·가격·thinking·API키)
- Ollama: https://ollama.com · GitHub https://github.com/ollama/ollama (로컬 LLM 실행·`/api`)
- 임베딩 모델 e5: https://huggingface.co/intfloat/multilingual-e5-small

## 3. 그래프 — Neo4j + GDS
- Neo4j 공식 문서: https://neo4j.com/docs
- Graph Data Science(GDS, 커뮤니티·PageRank): https://neo4j.com/docs/graph-data-science
- GDS GitHub: https://github.com/neo4j/graph-data-science
- Cypher(쿼리 언어): https://neo4j.com/docs/cypher-manual
- Docker 이미지: https://hub.docker.com/_/neo4j
- 핵심 알고리즘: `gds.louvain`(커뮤니티, L5), `gds.pageRank`(HippoRAG류)

## 4. 개발 도구
- uv(패키지·가상환경): https://docs.astral.sh/uv · GitHub https://github.com/astral-sh/uv
- Docker Compose: https://docs.docker.com/compose

## 5. 논문 · 공식 구현 (검색 방법의 진화)
| 기법 | 논문(arXiv) | 공식 구현 |
| --- | --- | --- |
| RAG (원조) | https://arxiv.org/abs/2005.11401 | HuggingFace Transformers `examples/rag` |
| **GraphRAG** | https://arxiv.org/abs/2404.16130 | https://github.com/microsoft/graphrag |
| CausalRAG | https://arxiv.org/abs/2503.19878 | https://github.com/Pwnb/CausalRAG |
| LightRAG | https://arxiv.org/abs/2410.05779 | https://github.com/HKUDS/LightRAG |
| HippoRAG / HippoRAG2 | https://arxiv.org/abs/2405.14831 · https://arxiv.org/abs/2502.14802 | https://github.com/OSU-NLP-Group/HippoRAG |
| LeanRAG | https://arxiv.org/abs/2508.10391 | https://github.com/RaZzzyz/LeanRAG |

## 6. 혼자 빌드 순서 (파일 → 문서 매핑)
```
1. loader/config/interface  →  LlamaIndex 시작(SimpleDirectoryReader·SentenceSplitter)
2. standard + eval 하니스    →  VectorStoreIndex        ← 여기까지면 RAG 이해 완료
3. bm25 → hybrid             →  BM25Retriever·QueryFusionRetriever
4. graphrag                  →  PropertyGraphIndex·*PathExtractor
5. 커스텀 추출기·재랭커      →  TransformComponent·BaseNodePostprocessor
6. Neo4j·커뮤니티요약        →  Neo4jPropertyGraphStore·GDS Louvain
```
> **2번(standard+eval)까지 혼자** 해보면 "RAG 짤 수 있다" 확신이 생김. 나머지는 그 위에 검색 방식만 교체.
