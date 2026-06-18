#!/usr/bin/env python3
"""
rank.py — CPU-only, no-network candidate ranker for the Redrob challenge.

    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

The semantic embeddings are PRE-COMPUTED offline (see precompute_embeddings.py)
and shipped as artifacts/. This script only loads them and does fast CPU scoring,
so it runs in well under the 5-minute / 16 GB / no-GPU / no-network budget.
"""
import argparse
import json
import numpy as np
import pandas as pd
from datetime import datetime

# ============================ knobs ============================
WEIGHTS = dict(sem=0.30, skill=0.28, title=0.22, exp=0.12, loc=0.05, edu=0.03)

CONSULTING_FIRMS = {
    "tcs", "tata consultancy services", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "tech mahindra", "hcl", "hcltech", "mindtree",
    "ltimindtree", "mphasis", "hexaware", "birlasoft", "dxc", "ntt data",
    "genpact", "larsen & toubro infotech",
}
CORE_SKILLS = {
    "embedding", "retrieval", "information retrieval", "vector database",
    "vector search", "faiss", "pinecone", "weaviate", "qdrant", "milvus",
    "elasticsearch", "opensearch", "semantic search", "hybrid search", "ranking",
    "learning to rank", "recommendation", "recommender", "search", "nlp",
    "natural language processing", "python", "bm25", "sentence-transformers",
    "sentence transformers", "bge", "e5", "ndcg", "mrr", "transformers", "llm",
    "large language model", "fine-tuning",
}
PROF_W = {"beginner": 0.4, "intermediate": 0.7, "advanced": 1.0, "expert": 1.0}
TARGET_TITLES = {"ai engineer", "machine learning", "ml engineer", "search engineer",
    "recommendation", "ranking", "retrieval", "research engineer", "nlp engineer",
    "applied scientist", "data scientist", "research scientist", "information retrieval"}
NEG_TITLES = {"operations", "customer support", "sales", "marketing", "hr ",
    "human resources", "recruiter", "accountant", "content writer", "business analyst",
    "project manager", "program manager", "qa ", "quality assurance", "support"}
INDIA_HUBS = {"pune", "noida", "hyderabad", "mumbai", "delhi", "gurgaon",
    "gurugram", "bangalore", "bengaluru", "ncr", "chennai"}


# ============================ helpers ============================
def parse_date(s):
    if not s:
        return None
    try:
        return datetime.strptime(str(s)[:10], "%Y-%m-%d").date()
    except Exception:
        return None

def is_consulting(name):
    n = (name or "").strip().lower()
    return any(f in n for f in CONSULTING_FIRMS)

def minmax(a):
    a = np.asarray(a, dtype=float)
    lo, hi = a.min(), a.max()
    return (a - lo) / (hi - lo) if hi > lo else a * 0.0

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ============================ feature extraction ============================
def extract_features(c, ref_date):
    p   = c.get("profile", {}) or {}
    ch  = c.get("career_history", []) or []
    edu = c.get("education", []) or []
    sk  = c.get("skills", []) or []
    sig = c.get("redrob_signals", {}) or {}

    durations = [j.get("duration_months", 0) or 0 for j in ch]
    companies = [(j.get("company") or "") for j in ch]
    starts    = [d for d in (parse_date(j.get("start_date")) for j in ch) if d]
    yoe       = float(p.get("years_of_experience") or 0)
    earliest  = min(starts) if starts else None
    span_yrs  = ((ref_date - earliest).days / 365.25) if earliest else 0.0
    n_expert_zero = sum(1 for s in sk if s.get("proficiency") in ("advanced", "expert")
                        and (s.get("duration_months", 0) or 0) == 0)
    last_active = parse_date(sig.get("last_active_date"))

    return {
        "candidate_id": c.get("candidate_id"),
        "current_title": p.get("current_title", ""),
        "country": p.get("country", ""),
        "years_of_experience": yoe,
        "num_jobs": len(ch),
        "avg_tenure_months": float(np.mean(durations)) if durations else 0.0,
        "exp_gap_years": round(abs(span_yrs - yoe), 2),
        "consulting_only": len(ch) > 0 and all(is_consulting(x) for x in companies),
        "skill_names": [(s.get("name") or "").lower() for s in sk],
        "n_expert_zero_dur": n_expert_zero,
        "edu_fields": [(e.get("field_of_study") or "").lower() for e in edu],
        "edu_top_tier": any(e.get("tier") in ("tier_1", "tier_2") for e in edu),
        "recruiter_response_rate": sig.get("recruiter_response_rate"),
        "days_since_active": (ref_date - last_active).days if last_active else None,
        "open_to_work": sig.get("open_to_work_flag"),
        "interview_completion_rate": sig.get("interview_completion_rate"),
        "saved_by_recruiters_30d": sig.get("saved_by_recruiters_30d"),
        "profile_completeness": sig.get("profile_completeness_score"),
        "willing_to_relocate": sig.get("willing_to_relocate", False),
    }


# ============================ component scores ============================
def core_skill_score(skills_raw):
    score = 0.0
    for s in skills_raw:
        name = (s.get("name") or "").lower()
        if any(k in name for k in CORE_SKILLS):
            prof  = PROF_W.get(s.get("proficiency"), 0.5)
            dur   = s.get("duration_months", 0) or 0
            end   = s.get("endorsements", 0) or 0
            trust = max(0.15, min(1.0, dur / 12) * 0.6 + min(1.0, end / 10) * 0.4)
            score += prof * trust
    return score

def title_score(titles):
    titles = [t.lower() for t in titles if t]
    cur = titles[0] if titles else ""
    cur_pos = any(k in cur for k in TARGET_TITLES)
    cur_neg = any(k in cur for k in NEG_TITLES)
    ever_pos = any(any(k in t for k in TARGET_TITLES) for t in titles)
    if cur_neg and not cur_pos:
        return 0.05
    return min(1.0, (0.7 if cur_pos else 0.0) + (0.3 if ever_pos else 0.0))

def loc_fit(country, location, relocate):
    if "india" in country.lower():
        return 1.0 if any(h in location.lower() for h in INDIA_HUBS) else 0.85
    return 0.5 if relocate else 0.25

def exp_fit(y):
    if 6 <= y <= 8:                  return 1.0
    if 5 <= y < 6 or 8 < y <= 9:     return 0.85
    if 4 <= y < 5 or 9 < y <= 11:    return 0.60
    if 3 <= y < 4 or 11 < y <= 14:   return 0.40
    return 0.20


# ============================ reasoning ============================
def matched_core(skill_names):
    out = []
    for s in skill_names:
        for k in CORE_SKILLS:
            if k in s and k not in out:
                out.append(k)
    return out[:3]

def make_reason(row):
    yrs, rk = float(row["years_of_experience"]), int(row["rank"])
    skills = matched_core(row["skill_names"])
    skill_txt = ", ".join(skills) if skills else "core ML skills"
    facts = [f"{row['current_title']}, {yrs:.1f} yrs in {row['country']}",
             f"matches on {skill_txt}"]
    extras = []
    if 6 <= yrs <= 8: extras.append("ideal experience band")
    elif yrs < 5:     extras.append("below the 6-8 yr ideal")
    rr = row["recruiter_response_rate"]
    if rr is not None and rr >= 0.7: extras.append("highly responsive")
    da = row["days_since_active"]
    if da is not None and da > 120:  extras.append("low recent activity")
    level = ("strong overall fit" if rk <= 20 else "good fit" if rk <= 50
             else "moderate fit" if rk <= 80 else "borderline fit")
    return "; ".join(facts + [level] + extras) + "."


# ============================ main ============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--cand-vecs", default="artifacts/cand_vecs.npy")
    ap.add_argument("--cand-ids",  default="artifacts/cand_ids.npy")
    ap.add_argument("--jd-vec",    default="artifacts/jd_vec.npy")
    args = ap.parse_args()

    print("Loading candidates ...")
    candidates = load_jsonl(args.candidates)

    ref_date = max(d for d in
                   (parse_date(c.get("redrob_signals", {}).get("last_active_date"))
                    for c in candidates) if d)

    df = pd.DataFrame(extract_features(c, ref_date) for c in candidates)

    print("Loading pre-computed embeddings ...")
    emb = np.load(args.cand_vecs).astype(np.float32)
    ids = np.load(args.cand_ids, allow_pickle=True)
    jd_vec = np.load(args.jd_vec).astype(np.float32)
    id2vec = {cid: emb[i] for i, cid in enumerate(ids)}
    cand_vecs = np.stack([id2vec[cid] for cid in df["candidate_id"]])
    df["semantic_score"] = cand_vecs @ jd_vec

    print("Scoring ...")
    raw_skills = [c.get("skills", []) or [] for c in candidates]
    df["skill_score"] = minmax([core_skill_score(s) for s in raw_skills])
    df["title_score"] = [title_score(
        [c.get("profile", {}).get("current_title", "")] +
        [j.get("title", "") for j in c.get("career_history", []) or []]) for c in candidates]
    df["loc_score"] = [loc_fit(r["country"],
                               candidates[i].get("profile", {}).get("location", ""),
                               r["willing_to_relocate"])
                       for i, r in df.iterrows()]
    df["exp_score"] = df["years_of_experience"].apply(exp_fit)

    CS = ("computer", "data", "artificial", "machine", "statistics", "electronics", "information")
    df["edu_score"] = (df["edu_top_tier"].astype(float) * 0.5 +
        df["edu_fields"].apply(lambda fs: 0.5 if any(any(k in f for k in CS) for f in fs) else 0.0))

    mod = ((0.8 + 0.4 * df["recruiter_response_rate"].fillna(0.5).clip(0, 1)) *
           (0.9 + 0.2 * df["interview_completion_rate"].fillna(0.5).clip(0, 1)) *
           (0.9 + 0.2 * (df["profile_completeness"].fillna(50) / 100).clip(0, 1)) *
           (1.0 + df["open_to_work"].fillna(False).astype(int) * 0.05) *
           (1.0 + (df["saved_by_recruiters_30d"].fillna(0) / 100).clip(0, 0.1)))
    recency = np.select(
        [df["days_since_active"].fillna(999) <= 30, df["days_since_active"].fillna(999) <= 90,
         df["days_since_active"].fillna(999) <= 180], [1.0, 0.95, 0.85], default=0.70)
    behavior_mod = mod * recency

    penalty = np.ones(len(df))
    penalty *= np.where(df["consulting_only"], 0.70, 1.0)
    penalty *= np.where((df["num_jobs"] >= 4) & (df["avg_tenure_months"] < 18), 0.80, 1.0)

    df["honeypot"] = (df["n_expert_zero_dur"] > 0) | (df["exp_gap_years"] > 3)
    hp_mult = np.where(df["honeypot"], 0.03, 1.0)

    base = (WEIGHTS["sem"] * minmax(df["semantic_score"]) + WEIGHTS["skill"] * df["skill_score"] +
            WEIGHTS["title"] * df["title_score"] + WEIGHTS["exp"] * df["exp_score"] +
            WEIGHTS["loc"] * df["loc_score"] + WEIGHTS["edu"] * df["edu_score"])
    df["final_score"] = (base * behavior_mod * penalty * hp_mult).round(6)

    top = df.sort_values(["final_score", "candidate_id"], ascending=[False, True]).head(100).reset_index(drop=True)
    top["rank"] = range(1, len(top) + 1)
    top["reasoning"] = top.apply(make_reason, axis=1)
    top["score"] = top["final_score"]

    out = top.sort_values(["score", "candidate_id"], ascending=[False, True]).reset_index(drop=True)
    out["rank"] = range(1, len(out) + 1)
    out[["candidate_id", "rank", "score", "reasoning"]].to_csv(args.out, index=False)
    print(f"Wrote {args.out} | rows: {len(out)} | honeypots in top: {int(top['honeypot'].sum())}")


if __name__ == "__main__":
    main()
