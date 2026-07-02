# your_project — RAG 방법론 비교·검증 벤치마크 (ragbench)

여러 RAG 기법(Standard/BM25/Hybrid/GraphRAG 계열)과 임베딩을 **같은 코퍼스·질문셋·조건**으로 비교하는 벤치마크. 문서 간 연결성 문제에 어떤 기법이 강한지 데이터로 검증한다.

- **핵심 결과**: 핀포인트 질의는 평면검색 우위, 전역·종합형 질의는 GraphRAG+커뮤니티요약 우위.
- **스택**: Python · LlamaIndex · 로컬 LLM(Ollama/Gemma) · 로컬 임베딩(e5) · Neo4j(GDS)
- **문서**: `CLAUDE.md`(설계·결과), `docs/WORKLOG.md`(타임라인)

> 코퍼스는 가상 회사 문서(주식회사 하울). 실제 데이터·개인정보 없음.
