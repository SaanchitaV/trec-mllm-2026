# TREC 2026 Million LLMs Track — LLM Expertise Ranking

Participant system for the [TREC 2026 Million LLMs Track](https://trec-mllm.github.io/):
given a user query and a pool of 150 LLMs, rank the LLMs by predicted expertise —
how likely each model is to produce a good answer — using only precomputed
"discovery" responses, without querying any model at run time.

The approach casts the task as **expertise retrieval / resource selection**:
each LLM is represented by the discovery responses it chose to answer
(≈84% of all responses are literal "I don't know" refusals, so answering
behaviour itself is a strong expertise signal), and ranking is done with
classical retrieval over these representations.

## Methods

| Method | Idea | Family |
|---|---|---|
| `profile_bm25` | Concatenate each LLM's substantive (query + answer) texts into one profile document; BM25 over 150 profiles | "Big-document" / candidate model (cf. Balog et al. Model 1; CORI-style resource selection) |
| `qknn` | BM25 over the 7,759 discovery queries; for a new query, credit the LLMs that substantively answered the top-30 most similar discovery queries | "Small-document" / document model (cf. Balog et al. Model 2; ReDDE-style) |
| `qknn_weighted` | As `qknn`, but refusals earn partial credit `1 − P(refusal)` computed from token logprobs — testing refusal *hesitation* as a graded expertise signal | Novel calibration-based variant (negative result) |
| `prior` | Rank by global answer count | Popularity baseline |
| `name_bm25` | Match query terms against LLM identifier names only | Zero-discovery-data floor |

## Development-set results (322 queries, full grid qrels, grades 0–2)

| Run | nDCG@10 | nDCG@5 | MRR | P@10 | MAP |
|---|---|---|---|---|---|
| **profile_bm25** | **0.2604** | 0.2292 | 0.3035 | 0.1478 | 0.2304 |
| qknn | 0.2023 | 0.1823 | 0.2533 | 0.1379 | 0.1870 |
| qknn_weighted | 0.1903 | 0.1795 | 0.2415 | 0.1339 | 0.1763 |
| name_bm25 (floor) | 0.1176 | 0.1057 | 0.1679 | 0.0894 | 0.1315 |
| prior | 0.0976 | 0.0933 | — | 0.1022 | 0.1122 |
| random (reference) | 0.103 | 0.089 | — | 0.092 | 0.119 |

Additional ablations (all below champion): thresholded calibration credit
(0.1914), champion+hesitation fusion at β=0.1/0.3 (0.2130 / 0.1775),
profile+qknn score fusion (0.2232). The calibration variants' consistent
failure suggests refusal confidence in these (visibly RAG-style) LLMs tracks
retrieval/context availability and query difficulty rather than graded domain
proximity.

A dense-retrieval variant (bge-small-en-v1.5 over per-response chunks,
top-3 chunk aggregation, and BM25+dense fusion) lives in
`scripts/dense_experiment.py` (GPU required).

## Repository layout

```
src/mllm_pipeline.py        build / rank / eval / test-run CLI (self-contained)
scripts/extract_logprobs.py streaming logprob summarizer for the 1.3 GB metadata file
scripts/dense_experiment.py dense retrieval + fusion experiment (GPU)
runs/                       dev-set run files for the main methods
docs/                       notes
```

## Data (not redistributed here)

All data comes from the track organizers and is **not** included in this
repository. Download from the official track page: https://trec-mllm.github.io/

- `llm_discovery_data.json` — 7,759 discovery queries × 150 LLM responses
- `llm_discovery_metadata.json` — same, with per-token logprobs (~1.3 GB)
- `llm_dev_data.tsv` — 322 development queries
- `llm_dev_qrels.txt` — full 322×150 graded qrels (0/1/2)
- Test topics — released Sep 5, 2026 per track timeline

The 2026 LLM pool is disjoint from 2025's; only methods transfer across years.

## Reproduce

```bash
pip install rank_bm25
python src/mllm_pipeline.py build --discovery llm_discovery_data.json --out structures.pkl
python src/mllm_pipeline.py rank  --queries llm_dev_data.tsv --method profile_bm25 \
       --structures structures.pkl --out run.dev.txt
python src/mllm_pipeline.py eval  --qrels llm_dev_qrels.txt --run run.dev.txt
# -> {'nDCG@10': 0.2604, ...}
```

Test-set submission (TREC six-column format):

```bash
python src/mllm_pipeline.py test-run --topics test_topics.jsonl \
       --method profile_bm25 --structures structures.pkl \
       --runid <team>_profile_bm25 --out run.test.txt
```

## Background reading

Resource selection / federated search: Callan (CORI, 1995); Si & Callan
(ReDDE, 2003); Aly et al. (Taily, 2013); Shokouhi & Si, *Federated Search*,
FnTIR. Expertise retrieval: Balog, Azzopardi & de Rijke (candidate vs.
document models); Balog et al., *Expertise Retrieval*, FnTIR 2012.
LLM routing and calibration: RouteLLM; FrugalGPT; RouterBench; Kadavath et
al., *Language Models (Mostly) Know What They Know* (2022).

## Author

Saanchita — TREC 2026 participant (Million LLMs + User Simulation tracks).
