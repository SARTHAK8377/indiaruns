# Deck outline (export to PDF) — tailored to what you built

~8-9 slides. Show the architecture diagram and a real slice of your top-10.

1. **Title** — "Ranking AI Engineers by fit, not keywords." Your name/team.

2. **The problem** — Keyword/ATS filters miss good people and get fooled by
   keyword-stuffers. The JD itself is full of "do NOT want" signals a keyword
   match can't read. Show the sample_submission's "Content Writer with 9 AI
   skills" as the trap.

3. **Data** — 100k candidates, deeply nested (career history, skills with
   proficiency/endorsements/duration, 23 behavioral signals). One fixed JD.

4. **Architecture** — the two-stage design: offline embedding pre-computation,
   then a CPU-only ranking step. Draw: candidates + JD -> features + embeddings
   -> hybrid score -> behavioral modifier -> penalties -> honeypot kill-switch
   -> top 100. Stress this respects the 5-min/CPU/no-network rule.

5. **Semantic layer** — embeddings give role relevance; show that pure semantic
   clusters scores at ~0.90 and ignores experience, location, honeypots.

6. **Structured layer** — the 6 components + weights. Emphasize the *title*
   component as the keyword-stuffer defense, and the endorsement+duration trust
   weighting on skills.

7. **Behavioral + honeypots** — availability modifier (response rate, recency,
   recruiter saves) and the consistency checks that force ~impossible profiles
   out. Report: 0 honeypots in top 100.

8. **Results** — your top-10 table with reasoning. Call out a before/after:
   the 16.9-yr "AI Engineer" semantic loved, dropped by the structured layer.

9. **Why it's production-ready** — runs CPU-only in minutes, scales to 200k,
   every rank carries an explanation. Limits + next steps (learn-to-rank,
   bias checks, feedback loop).
