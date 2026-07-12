"""로컬 e5를 OpenAI 호환 임베딩 API로 서빙 — HippoRAG 공정비교용.

다른 기법과 **완전 동일 벡터**를 쓰도록 LlamaIndex HuggingFaceEmbedding(우리 기본 e5)을
그대로 감싼다. HippoRAG 는 embedding_model_name 에 "text-embedding" 이 들어가면
OpenAI 임베딩 클래스로 라우팅 → 이 서버의 /v1/embeddings 를 호출한다.

실행: .venv/bin/python scripts/e5_openai_server.py  [--port 8123] [--model intfloat/multilingual-e5-small]
"""
from __future__ import annotations

import argparse
from typing import List, Union

import torch
import uvicorn
from fastapi import FastAPI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pydantic import BaseModel

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=8123)
parser.add_argument("--model", default="intfloat/multilingual-e5-small")
args, _ = parser.parse_known_args()

_device = "cuda" if torch.cuda.is_available() else "cpu"
_emb = HuggingFaceEmbedding(model_name=args.model, device=_device, embed_batch_size=8)
app = FastAPI()


class EmbReq(BaseModel):
    input: Union[str, List[str]]
    model: str | None = None


@app.get("/health")
def health():
    return {"status": "ok", "model": args.model, "device": _device}


@app.post("/v1/embeddings")
def embeddings(req: EmbReq):
    texts = [req.input] if isinstance(req.input, str) else list(req.input)
    vecs = _emb.get_text_embedding_batch(texts)  # 우리 다른 기법과 동일 경로
    data = [{"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(vecs)]
    return {"object": "list", "data": data, "model": req.model or args.model,
            "usage": {"prompt_tokens": 0, "total_tokens": 0}}


if __name__ == "__main__":
    print(f"e5 OpenAI호환 서버: http://127.0.0.1:{args.port}/v1  (model={args.model}, {_device})")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
