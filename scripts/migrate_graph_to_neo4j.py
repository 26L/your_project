"""로컬 그래프(SimplePropertyGraphStore, storage/<method>/)를 Neo4j로 적재.

재추출 없이 이미 만든 그래프를 Neo4j에 넣어 Cypher/GDS/브라우저 시각화를 쓰기 위함.
사전: `docker compose up -d` 로 Neo4j 가동(docker-compose.yml).

사용:
    .venv/bin/python scripts/migrate_graph_to_neo4j.py [storage/graphrag_e2b]
"""
from __future__ import annotations

import os
import sys

from llama_index.core.graph_stores.simple_labelled import SimplePropertyGraphStore
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore

PERSIST = sys.argv[1] if len(sys.argv) > 1 else "storage/graphrag_e2b"
URL = os.environ.get("NEO4J_URL", "bolt://localhost:7687")
USER = os.environ.get("NEO4J_USER", "neo4j")
PWD = os.environ.get("NEO4J_PASSWORD", "ragbench123")


def main() -> None:
    src = SimplePropertyGraphStore.from_persist_dir(PERSIST)
    nodes = list(src.graph.nodes.values())
    rels = list(src.graph.relations.values())
    print(f"적재 대상({PERSIST}): 노드 {len(nodes)} · 관계 {len(rels)}")

    neo = Neo4jPropertyGraphStore(username=USER, password=PWD, url=URL)
    neo.structured_query("MATCH (n) DETACH DELETE n")  # 초기화
    neo.upsert_nodes(nodes)
    neo.upsert_relations(rels)

    c = neo.structured_query("MATCH (n) RETURN count(n) AS c")[0]["c"]
    r = neo.structured_query("MATCH ()-[x]->() RETURN count(x) AS c")[0]["c"]
    print(f"Neo4j 적재 완료: 노드 {c} · 관계 {r}  (브라우저: http://localhost:7474)")


if __name__ == "__main__":
    main()
