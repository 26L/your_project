"""ragbench CLI — 인덱싱/질의 진입점.

  ragbench index --method standard --data data
  ragbench query --method standard "질문 내용"
"""
from __future__ import annotations

import argparse

from dotenv import load_dotenv

from .core.config import Config
from .embeddings.factory import build_embed_model
from .ingest.loader import load_documents
from .llms.factory import build_llm
from .registry import build_backend


def _make_backend(args):
    cfg = Config.load(args.config)
    llm = build_llm(cfg)
    embed = build_embed_model(cfg)
    return cfg, build_backend(args.method, cfg, llm, embed)


def cmd_index(args):
    cfg, backend = _make_backend(args)
    data_dir = args.data or cfg.data_dir
    docs = load_documents(data_dir)
    print(f"'{data_dir}' 에서 문서 {len(docs)}개 로드. '{args.method}' 기법으로 인덱싱 중...")
    backend.index(docs)
    print(f"인덱스 저장 완료 → {cfg.storage_dir}/{args.method}")


def cmd_query(args):
    _, backend = _make_backend(args)
    result = backend.query(args.question)
    print("\n=== 답변 ===")
    print(result.answer)
    print("\n=== 출처 ===")
    for i, ctx in enumerate(result.contexts, 1):
        print(f"[{i}] {ctx.source}  (score={ctx.score})")


def _fmt_agg(agg: dict) -> str:
    keys = ("n", "recall", "precision", "mrr", "keyword_recall", "judge_correct", "latency_s")
    return "  ".join(f"{k}={agg.get(k)}" for k in keys if agg.get(k) is not None)


def cmd_eval(args):
    import json
    import os

    from .eval.dataset import load_eval_set
    from .eval.harness import run_eval

    cfg, backend = _make_backend(args)
    judge_llm = build_llm(cfg) if args.judge else None
    items = load_eval_set(args.eval_set)
    print(
        f"평가셋 {len(items)}문항 · 기법 '{args.method}' · top_k={cfg.top_k}"
        f"{' · LLM-judge ON' if args.judge else ''} 실행 중..."
    )
    report = run_eval(backend, items, cfg.top_k, judge_llm=judge_llm)

    print("\n=== 전체 집계 ===")
    print(" ", _fmt_agg(report["aggregate"]))
    print("\n=== 유형별 ===")
    for qtype, agg in report["by_type"].items():
        print(f"  [{qtype:10s}] {_fmt_agg(agg)}")

    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", f"{args.method}.json")
    payload = {
        "method": args.method,
        "config": {"llm": cfg.llm.model, "embed": cfg.embed.model, "top_k": cfg.top_k},
        "report": report,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장 → {out_path}")


def cmd_compare(args):
    import glob
    import json

    paths = sorted(glob.glob("results/*.json"))
    if not paths:
        print("results/ 에 결과 파일이 없습니다. 먼저 'ragbench eval' 를 실행하세요.")
        return

    import os

    rows = []
    for p in paths:
        d = json.load(open(p, encoding="utf-8"))
        label = os.path.splitext(os.path.basename(p))[0]  # 파일명 기준 라벨
        rows.append((label, d["report"]["aggregate"]))

    cols = ("recall", "precision", "mrr", "keyword_recall", "judge_correct", "latency_s")
    header = f"{'method':12s} " + " ".join(f"{c:>14s}" for c in cols)
    print(header)
    print("-" * len(header))
    for method, agg in rows:
        cells = " ".join(f"{str(agg.get(c)):>14s}" for c in cols)
        print(f"{method:12s} {cells}")


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(prog="ragbench")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default="config/default.yaml")
    common.add_argument("--method", default="standard")

    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", parents=[common], help="문서 인덱싱")
    p_index.add_argument("--data", help="문서 디렉토리 (기본: config 의 data_dir)")
    p_index.set_defaults(func=cmd_index)

    p_query = sub.add_parser("query", parents=[common], help="질의")
    p_query.add_argument("question", help="질문 텍스트")
    p_query.set_defaults(func=cmd_query)

    p_eval = sub.add_parser("eval", parents=[common], help="평가셋으로 지표 집계")
    p_eval.add_argument("--eval-set", default="config/eval_sample.yaml", help="평가셋 YAML")
    p_eval.add_argument("--judge", action="store_true", help="LLM-as-judge 정답 채점(비용 발생)")
    p_eval.set_defaults(func=cmd_eval)

    p_compare = sub.add_parser("compare", help="results/*.json 기법 비교표")
    p_compare.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
