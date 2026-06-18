# Redrob Candidate Ranker

Ranks the top 100 candidates for the *Senior AI Engineer (Founding Team)* role
from a 100,000-candidate pool — by understanding role fit, not matching keywords.

## Reproduce (the single command)

```bash
pip install -r requirements.txt
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

This runs **CPU-only, no network, in well under 5 minutes** on 16 GB RAM. It
loads the pre-computed embeddings in `artifacts/` and does fast structured
scoring — no LLM calls, no GPU.

## Two-stage design (respects the compute rules)

**Pre-computation (offline, GPU + network allowed):**
```bash
python precompute_embeddings.py --candidates ./candidates.jsonl --out-dir ./artifacts
```
Encodes each candidate profile and the job description with `BAAI/bge-small-en-v1.5`
and saves them as float16 `.npy` files (~77 MB, fits in the repo). Run once.

**Ranking (the scored step, CPU-only, no network):** `rank.py` loads those vectors
and produces the ranking. This is the part reproduced in the Stage-3 sandbox.

## How candidates are scored

A weighted hybrid score, then a behavioral modifier, penalties, and a honeypot
kill-switch:

```
base    = 0.30*semantic + 0.28*skills + 0.22*title + 0.12*experience
        + 0.05*location + 0.03*education
final   = base * behavioral_modifier * negative_penalties * honeypot_killswitch
```

- **semantic** — cosine similarity of profile vs. a distilled JD query (embeddings).
- **skills** — overlap with the JD's core skills (embeddings/retrieval, vector DBs,
  ranking, NLP, Python...), trust-weighted by endorsements + duration so
  keyword-stuffed skills with 0 months of use barely count.
- **title** — current/past titles vs. target roles; an off-role current title
  (e.g. "Operations Manager") is forced near zero. This is the main defense
  against keyword-stuffer traps.
- **experience** — peaks at the JD's ideal 6–8 years, tapers outside 5–9.
- **location** — India hubs preferred; outside India down-weighted (no visa sponsorship).
- **behavioral modifier** — recruiter response rate, recency, interview completion,
  recruiter saves, profile completeness, open-to-work. A perfect-on-paper but
  inactive/unresponsive candidate is effectively unavailable.
- **negative penalties** — consulting-only careers and short-tenure job-hoppers
  (both explicitly unwanted in the JD).
- **honeypot kill-switch** — impossible profiles (expert skills with 0 months used;
  claimed experience far exceeding actual career span) are multiplied by 0.03 so
  they cannot reach the top 100.

## Files

| File | Purpose |
|------|---------|
| `rank.py` | the scored ranking step (CPU, no network) |
| `precompute_embeddings.py` | offline embedding generation |
| `artifacts/` | pre-computed `cand_vecs.npy`, `cand_ids.npy`, `jd_vec.npy` |
| `submission_metadata.yaml` | challenge submission metadata |
| `requirements.txt` | dependencies |

## Output format

`submission.csv` — header `candidate_id,rank,score,reasoning`, exactly 100 rows,
ranks 1–100, score non-increasing, ties broken by candidate_id ascending.
Validated with the provided `validate_submission.py`.
