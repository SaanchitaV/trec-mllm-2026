"""Stream llm_discovery_metadata.json (~1.3 GB) into a compact per-response
summary CSV: qid,llm,is_idk,p_response,mean_logprob,n_tokens.
Usage: python scripts/extract_logprobs.py llm_discovery_metadata.json > logprob_summary.csv
"""
import ijson, math, sys

print("qid,llm,is_idk,p_response,mean_logprob,n_tokens")
with open(sys.argv[1], "rb") as f:
    for qid, rec in ijson.kvitems(f, ""):
        for llm, r in rec["responses"].items():
            lps = [float(x) for x in r.get("logprobs", [])]
            if not lps:
                continue
            is_idk = int(r["response"].strip().lower().startswith("i don't know"))
            total = sum(lps)
            p = math.exp(total) if is_idk else 0.0
            print(f"{qid},{llm},{is_idk},{p:.6f},{total/len(lps):.5f},{len(lps)}")
