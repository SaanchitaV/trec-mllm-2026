"""TREC 2026 Million LLMs — full pipeline in one file.

Usage:
  python mllm_pipeline.py build      --discovery llm_discovery_data.json
  python mllm_pipeline.py rank       --queries llm_dev_data.tsv --method profile_bm25 --out run.dev.txt
  python mllm_pipeline.py eval       --qrels llm_dev_qrels.txt --run run.dev.txt
  python mllm_pipeline.py test-run   --topics test_topics.jsonl --method profile_bm25 --runid <team>_bm25 --out run.test.txt

Methods: profile_bm25 (champion), qknn, prior, name_bm25.
Dev scores (322 queries, full discovery data, Aug 2026):
  profile_bm25 nDCG@10=0.260 MRR=0.304 | qknn 0.202/0.253 | name floor 0.118/0.168
"""
import argparse, json, math, pickle, re, sys
from collections import defaultdict

TOK = re.compile(r"[a-z0-9]+")
def tok(s): return TOK.findall(s.lower())
def is_idk(r): return r.strip().lower().startswith("i don't know")


# --------------------------------------------------------------------- build
def build(discovery_path, out="structures.pkl", answer_cap=120):
    data = json.load(open(discovery_path))
    profiles, count = defaultdict(list), defaultdict(int)
    dq_tokens, dq_llms = [], []
    for qid, rec in data.items():
        qt = tok(rec["query"]); answerers = []
        for llm, resp in rec["responses"].items():
            if not is_idk(resp):
                count[llm] += 1
                profiles[llm].extend(qt); profiles[llm].extend(tok(resp)[:answer_cap])
                answerers.append(llm)
        dq_tokens.append(qt); dq_llms.append(answerers)
    pickle.dump({"profiles": dict(profiles), "count": dict(count),
                 "dq_tokens": dq_tokens, "dq_llms": dq_llms}, open(out, "wb"))
    print(f"built {out}: {len(profiles)} LLMs, {len(dq_tokens)} discovery queries")


# --------------------------------------------------------------------- rank
def _load_queries(path):
    if path.endswith(".jsonl"):
        return [(r["topic_id"], r["question"]) for r in map(json.loads, open(path))]
    return [tuple(l.rstrip("\n").split("\t")) for l in open(path)]

def rank(queries_path, method, out, runid=None, structures="structures.pkl", k=30):
    from rank_bm25 import BM25Okapi
    S = pickle.load(open(structures, "rb"))
    llms = sorted(S["profiles"]); runid = runid or method
    queries = _load_queries(queries_path)
    if method == "profile_bm25":
        bm = BM25Okapi([S["profiles"][l] for l in llms])
        scorer = lambda q: dict(zip(llms, bm.get_scores(tok(q))))
    elif method == "qknn":
        qbm = BM25Okapi(S["dq_tokens"])
        def scorer(q):
            sims = qbm.get_scores(tok(q))
            top = sorted(range(len(sims)), key=lambda i: -sims[i])[:k]
            sc = defaultdict(float)
            for i in top:
                if sims[i] > 0:
                    for l in S["dq_llms"][i]: sc[l] += sims[i]
            return sc
    elif method == "prior":
        scorer = lambda q: S["count"]
    elif method == "name_bm25":
        docs = [[w[:-1] if w.endswith("s") and len(w) > 3 else w
                 for w in tok(l.replace("_", " "))] for l in llms]
        bm = BM25Okapi(docs)
        scorer = lambda q: dict(zip(llms, bm.get_scores(
            [w[:-1] if w.endswith("s") and len(w) > 3 else w for w in tok(q)])))
    else:
        sys.exit(f"unknown method {method}")
    with open(out, "w") as f:
        for qid, q in queries:
            sc = scorer(q)
            ranked = sorted(((l, sc.get(l, 0.0)) for l in llms), key=lambda x: -x[1])
            for r, (l, s) in enumerate(ranked, 1):
                f.write(f"{qid} Q0 {l} {r} {s:.4f} {runid}\n")
    print(f"wrote {out} ({len(queries)} queries x {len(llms)} LLMs, runid={runid})")


# --------------------------------------------------------------------- eval
def load_qrels(path):
    q = defaultdict(dict)
    for line in open(path):
        p = line.split()
        if len(p) >= 4: q[p[0]][p[2]] = int(p[3])
    return q

def evaluate(qrels_path, run_path):
    qrels = load_qrels(qrels_path)
    run = defaultdict(list)
    for line in open(run_path):
        p = line.split(); run[p[0]].append((float(p[4]), p[2]))
    run = {q: [d for _, d in sorted(v, key=lambda x: -x[0])] for q, v in run.items()}
    n10 = n5 = p10 = ap_sum = mrr = 0.0; n = 0
    for qid, rels in qrels.items():
        ranked = run.get(qid, []); n += 1
        for K, acc in ((10, "n10"), (5, "n5")): pass
        def ndcg(k):
            dcg = sum((2**rels.get(d, 0) - 1) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
            ideal = sorted(rels.values(), reverse=True)[:k]
            idcg = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
            return dcg / idcg if idcg else 0.0
        n10 += ndcg(10); n5 += ndcg(5)
        p10 += sum(rels.get(d, 0) > 0 for d in ranked[:10]) / 10
        hits = 0; ap = 0.0; R = sum(g > 0 for g in rels.values())
        for i, d in enumerate(ranked):
            if rels.get(d, 0) > 0:
                hits += 1; ap += hits / (i + 1)
                if hits == 1: mrr += 1 / (i + 1)
        ap_sum += ap / R if R else 0.0
    res = {"nDCG@10": round(n10/n, 4), "nDCG@5": round(n5/n, 4), "MRR": round(mrr/n, 4),
           "P@10": round(p10/n, 4), "MAP": round(ap_sum/n, 4), "queries": n}
    print(res); return res


# ---------------------------------------------------------------------- cli
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--discovery", required=True)
    b.add_argument("--out", default="structures.pkl")
    r = sub.add_parser("rank"); r.add_argument("--queries", required=True)
    r.add_argument("--method", default="profile_bm25"); r.add_argument("--out", required=True)
    r.add_argument("--structures", default="structures.pkl"); r.add_argument("--runid")
    t = sub.add_parser("test-run"); t.add_argument("--topics", required=True)
    t.add_argument("--method", default="profile_bm25"); t.add_argument("--out", required=True)
    t.add_argument("--structures", default="structures.pkl"); t.add_argument("--runid", required=True)
    e = sub.add_parser("eval"); e.add_argument("--qrels", required=True); e.add_argument("--run", required=True)
    a = ap.parse_args()
    if a.cmd == "build": build(a.discovery, a.out)
    elif a.cmd in ("rank", "test-run"):
        rank(a.topics if a.cmd == "test-run" else a.queries, a.method, a.out,
             runid=a.runid, structures=a.structures)
    elif a.cmd == "eval": evaluate(a.qrels, a.run)
