"""문서 로드 (모든 기법이 공유). 청킹은 기법별 인덱싱 단계에서 설정한다."""
from __future__ import annotations

from typing import Any


def load_documents(data_dir: str) -> list[Any]:
    """data_dir 의 파일들을 LlamaIndex Document 리스트로 읽는다."""
    from llama_index.core import SimpleDirectoryReader

    return SimpleDirectoryReader(data_dir, recursive=True).load_data()
