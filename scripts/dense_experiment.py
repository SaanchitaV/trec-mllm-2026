"""Dense retrieval + fusion experiment (GPU recommended).
Chunks = "discovery query | substantive answer[:600]"; LLM score = sum of its
top-3 chunk cosine similarities (small-document aggregation). Then fuse with
the BM25 champion at w in {0.3, 0.5, 0.7}.

Usage (after `build` and a profile_bm25 dev run exist):
  python scripts/dense_experiment.py \
      --discovery llm_discovery_data.json --dev llm_dev_data.tsv \
      --qrels llm_dev_qrels.txt --bm25-run run.dev.txt --emb-cache chunk_emb.npy
"""
import argparse, json
from collections import defaultdict
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    for k in ("discovery", "dev", "qrels", "bm25_run"):
        ap.add_argument("--" + k.replace("_", "-"), required=True)
    ap.add_argument("--emb-cache", default="chunk_emb.npy")
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    a = ap.parse_args()

    from sentence_transformers import SentenceTransformer
    import os, subprocess, sys

    data = json.load(open(a.discovery))
    chunks, chunk_llm = [], []
    for qid, rec in data.items():
        q = rec["query"]
        for llm, resp in rec["responses"].items():
            if not resp.strip().lower().startswith("i don't know"):
                chunks.append(q + " | " + resp[:600]); chunk_llm.append(llm)
    llms = sorted(set(chunk_llm))
    idx = {l: i for i, l in enumerate(llms)}
    cli = np.array([idx[l] for l in chunk_llm])
    print(len(chunks), "chunks,", len(llms), "LLMs")

    model = SentenceTransformer(a.model)
    if os.path.exists(a.emb_cache):
        emb = np.load(a.emb_cache)
    else:
        emb = model.encode(chunks, batch_size=256, normalize_embeddings=True,
                           show_progress_bar=True, convert_to_numpy=True).astype("float32")
        np.save(a.emb_cache, emb)

    dev = [l.rstrip("\n").split("\t") for l in open(a.dev)]
    qs = ["Represent this sentence for searching relevant passages: " + q for _, q in dev]
    qemb = model.encode(qs, batch_size=64, normalize_embeddings=True,
                        convert_to_numpy=True).astype("float32")

    with open("run.dense.txt", "w") as f:
        for (qid, _), qv in zip(dev, qemb):
            sims = emb @ qv
            scores = np.zeros(len(llms))
            for li in range(len(llms)):
                s = sims[cli == li]
                if len(s): scores[li] = np.sort(s)[-3:].sum()
            for r, li in enumerate(np.argsort(-scores), 1):
                f.write(f"{qid} Q0 {llms[li]} {r} {scores[li]:.4f} dense\n")

    def load(p):
        r = defaultdict(dict)
        for line in open(p):
            q, _, l, _, s, _ = line.split(); r[q][l] = float(s)
        return r
    A, B = load(a.bm25_run), load("run.dense.txt")
    for w in (0.3, 0.5, 0.7):
        with open(f"run.fuse{w}.txt", "w") as f:
            for q in A:
                am = max(A[q].values()) or 1; bm = max(B[q].values()) or 1
                sc = {l: (1-w)*A[q][l]/am + w*B[q].get(l, 0)/bm for l in A[q]}
                for r, (l, s) in enumerate(sorted(sc.items(), key=lambda x: -x[1]), 1):
                    f.write(f"{q} Q0 {l} {r} {s:.4f} fuse{w}\n")
    print("wrote run.dense.txt, run.fuse{0.3,0.5,0.7}.txt — evaluate with mllm_pipeline.py eval")

if __name__ == "__main__":
    main()
