"""Decode benchmark for the local sglang server.

Measures per-case decode throughput (greedy, fixed reasoning effort) against
whichever server is currently running on port 30000. The model id is
auto-detected, so the same command works for every run*.bat.

Usage:
  python bench.py --label dflash2-baseline
  python bench.py --cases py-codegen,longctx-13k --passes 5
  python bench.py --list

Method per case: flush server cache -> one max_tokens=1 call (true prefill)
-> N full passes (prefix now radix-cached, so timing is decode-dominated).
Pass-to-pass tok/s spread also exposes dictionary/cache warmup effects
(e.g. NGRAM or the fusion prototype speed up on later passes).
"""

import argparse
import datetime
import json
import pathlib
import statistics
import sys
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent
LONG_SOURCE = ROOT / "patches" / "loader.py"


def build_cases():
    long_code = LONG_SOURCE.read_text(encoding="utf-8")
    mid_code = long_code[:16000]
    return [
        ("py-codegen", "Write a Python implementation of a thread-safe LRU cache with TTL support, with docstrings and a small usage example.", 800),
        ("ts-codegen", "Write a TypeScript function that parses a cron expression into its five fields, validates ranges, and returns a structured object. Include error handling and unit tests.", 800),
        ("sql", "Design a PostgreSQL schema for a multi-tenant SaaS billing system (tenants, plans, subscriptions, invoices, usage records), then write one analytical query that computes MRR per tenant per month with proration. Explain the query briefly.", 600),
        ("json-out", "Extract the following into strict JSON with keys name, version, deps (array), scripts (object): 'The project foo-cli v2.3.1 depends on chalk, commander and zod. It has scripts: build runs tsc, test runs vitest run, lint runs eslint src.' Output only JSON.", 300),
        ("prose-ja", "投機的デコード（speculative decoding）の仕組みを、draft/verifyの流れ、受理率、なぜ出力品質が劣化しないのかを含めて、日本語で技術ブログ風に説明してください。", 700),
        ("math", "A factory produces widgets on 3 lines. Line A makes 120/hr with 2% defects, B makes 200/hr with 3.5% defects, C makes 150/hr with 1.5% defects. All run 16 hours/day. Compute total good widgets per day, overall defect rate, and how many hours line B alone would need to replace one full day of C's good output. Show your work.", 600),
        ("boilerplate", "Generate Pydantic v2 models and FastAPI CRUD endpoints (create/read/update/delete/list with pagination) for entities User, Team, and Project. Follow the exact same structure for each entity.", 1400),
        ("midctx-4k", "Review the following Python code. List concrete bugs or risky patterns with line references, then suggest the three highest-impact refactorings:\n\n```python\n" + mid_code + "\n```", 600),
        ("longctx-13k", "Summarize what this Python file does in 5 bullet points, then list its main classes with one line each:\n\n```python\n" + long_code[:60000] + "\n```", 400),
    ]


def http(base, path, payload=None, timeout=600):
    data = json.dumps(payload).encode() if payload is not None else b""
    req = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def detect_model(base):
    with urllib.request.urlopen(base + "/v1/models", timeout=30) as r:
        return json.load(r)["data"][0]["id"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:30000")
    ap.add_argument("--label", default=None, help="run label (default: <model-tail>-<timestamp>)")
    ap.add_argument("--out", default=None, help="output jsonl (default: results/<label>.jsonl)")
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--cases", default="all", help="comma-separated case names, or 'all'")
    ap.add_argument("--reasoning-effort", default="low")
    ap.add_argument("--list", action="store_true", help="list case names and exit")
    args = ap.parse_args()

    cases = build_cases()
    if args.list:
        for name, _, max_tok in cases:
            print(f"{name}  (max_tokens={max_tok})")
        return

    if args.cases != "all":
        wanted = set(args.cases.split(","))
        unknown = wanted - {n for n, _, _ in cases}
        if unknown:
            sys.exit(f"unknown cases: {sorted(unknown)}")
        cases = [c for c in cases if c[0] in wanted]

    model = detect_model(args.url)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    label = args.label or f"{model.split('/')[-1]}-{stamp}"
    out = pathlib.Path(args.out) if args.out else ROOT / "results" / f"{label}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    def chat(content, max_tokens):
        body = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "chat_template_kwargs": {"reasoning_effort": args.reasoning_effort},
        }
        t0 = time.time()
        r = json.load(http(args.url, "/v1/chat/completions", body))
        return time.time() - t0, r["usage"]

    print(f"# label={label} model={model} passes={args.passes}")
    chat("Say OK.", 8)  # warmup

    results = []
    for name, content, max_tok in cases:
        try:
            http(args.url, "/flush_cache")
        except Exception:
            pass
        t_pre, u_pre = chat(content, 1)
        times, gens = [], []
        for _ in range(args.passes):
            dt, u = chat(content, max_tok)
            times.append(dt)
            gens.append(u["completion_tokens"])
        tps = [g / t for g, t in zip(gens, times)]
        rec = {
            "label": label,
            "model": model,
            "timestamp": stamp,
            "case": name,
            "prompt_tokens": u_pre["prompt_tokens"],
            "prefill_s": round(t_pre, 3),
            "gen_tokens": gens,
            "pass_s": [round(t, 3) for t in times],
            "tok_s": [round(x, 1) for x in tps],
            "tok_s_mean": round(statistics.mean(tps), 1),
        }
        results.append(rec)
        print(f"{name:12s} prompt={rec['prompt_tokens']:>6}t prefill={rec['prefill_s']:>6.2f}s "
              f"tok/s={rec['tok_s']} mean={rec['tok_s_mean']}")

    with out.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
