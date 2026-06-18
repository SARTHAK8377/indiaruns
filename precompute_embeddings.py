#!/usr/bin/env python3
"""
precompute_embeddings.py — OFFLINE pre-computation (GPU + network allowed here).

    python precompute_embeddings.py --candidates ./candidates.jsonl --out-dir ./artifacts

Produces artifacts/cand_vecs.npy (float16), artifacts/cand_ids.npy, artifacts/jd_vec.npy.
This step may exceed the 5-minute budget and use a GPU — that is permitted because
it is pre-computation. The ranking step (rank.py) only LOADS these files and runs
CPU-only with no network, well within the limits.
"""
import argparse
import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL = "BAAI/bge-small-en-v1.5"

JD_QUERY = (
    "Senior AI / Machine Learning Engineer for a production ranking and "
    "retrieval system. Must have production experience with embeddings-based "
    "retrieval (sentence-transformers, BGE, E5, OpenAI embeddings), vector "
    "databases and hybrid search (FAISS, Pinecone, Weaviate, Qdrant, Milvus, "
    "Elasticsearch, OpenSearch), and strong Python. Designs evaluation "
    "frameworks for ranking systems (NDCG, MRR, MAP, A/B testing). Has shipped "
    "an end-to-end search, ranking, or recommendation system to real users at "
    "scale. Strong NLP and information retrieval background. 6-8 years of "
    "applied ML at product companies."
)


def profile_text(c):
    p = c.get("profile", {}) or {}
    sk = c.get("skills", []) or []
    parts = [p.get("headline", ""), p.get("summary", ""),
             "Skills: " + ", ".join(s.get("name", "") for s in sk)]
    for j in c.get("career_history", []) or []:
        parts.append(f"{j.get('title','')} at {j.get('company','')}: {j.get('description','')}")
    return "\n".join(x for x in parts if x).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out-dir", default="artifacts")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    candidates = []
    with open(args.candidates, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))

    model = SentenceTransformer(MODEL)  # uses GPU automatically if available

    jd_vec = model.encode(
        "Represent this sentence for searching relevant passages: " + JD_QUERY,
        normalize_embeddings=True).astype("float32")

    texts = [profile_text(c) for c in candidates]
    cand_vecs = model.encode(texts, batch_size=256, normalize_embeddings=True,
                             show_progress_bar=True).astype("float16")  # half precision -> <100MB
    cand_ids = np.array([c["candidate_id"] for c in candidates])

    np.save(os.path.join(args.out_dir, "cand_vecs.npy"), cand_vecs)
    np.save(os.path.join(args.out_dir, "cand_ids.npy"), cand_ids)
    np.save(os.path.join(args.out_dir, "jd_vec.npy"), jd_vec)
    print(f"Saved artifacts to {args.out_dir}/  shape={cand_vecs.shape}")


if __name__ == "__main__":
    main()
