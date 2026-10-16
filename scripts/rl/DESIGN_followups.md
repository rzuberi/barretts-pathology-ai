# RL follow-ups A–F: DESIGN (fixed 2026-10-16 17:00, before any run)

Shared rules of PROJECT_BRIEFS.md apply: report text stays on the cluster; the 479 graded human sections are test only; patient
bootstrap (2,000) for every CI; checkpoints chosen on validation only; smoke first; race-to-run; git author rzuberi.

## A. Calibration (CPU, existing outputs only)
Scores and the system each one judges: (a) jury vote fraction → jury consensus grade; (b) project-1 RL model grade-token probability
→ its grade; (c) its 8-sample agreement → its grade; (d) project-2 α=0.2 model grade-token probability → its grade.
"Correct" = system worst grade == Rehan's worst grade. Sets: the 479 sections (population-weighted) and the n=50 random slice (unweighted).
1. Error-detection AUROC (weighted Mann–Whitney: a low score should mark an error). 2. ECE, 10 equal-width bins, + reliability diagram,
for (a) and (b). 3. Risk–coverage curves (weighted error among the most confident fraction kept) for all four, with the operating
point of jury triage at 20% of reports (= 13% of sections reviewed, 87% coverage) marked. 4. Primary: ΔAUROC (b) − (a), paired.
Note: the grade-token probability is that of the first token of the grade value.

## B. Second rater (blocked until Jonathan grades; script ready)
Rehan vs Jonathan on co-graded sections: exact and dysplasia agreement, quadratic-weighted kappa on the ordinal scale, confusion.
Jury, RL model, MedGemma-27B against each rater and against the sections where both agree. Weighted and unweighted; partial data
reported as partial with n.

## C. Seeds (≈15 GPU-h)
Seeds 1 and 2 added to seed 0 for SFT and SFT→GRPO, everything else identical to project 1 (seed sets LoRA init, data order and
sampling). Each GRPO run's checkpoint chosen on validation. Report mean and range over 3 seeds of weighted exact, dysplasia yes/no
and RL − SFT (same seed). Paired best-seed and seed-mean RL vs MedGemma-27B on the 479.
**Pre-registered question: is RL − SFT > 0 in all three seeds?**

## D. Fixing the abstention reward (≈10 GPU-h)
Both from the project-1 RL model, 300 GRPO steps, project-1 hyperparameters, training prompts = class-capped training sections
(≤1,200 per class) PLUS all 1,028 low-agreement training-patient sections (not capped).
- D1 jury-conditioned abstention: vote fraction < 0.75 → UNSURE +0.5, grade == consensus +1, other grade −1; vote fraction ≥ 0.75 →
  UNSURE 0, correct +1, wrong −1; invalid −1. Triage order: UNSURE first, then grade-token probability ascending.
- D2 stated confidence: output {"grade", "confidence": 0–100}; reward = 1 − (confidence/100 − correct)², invalid −1.
  Triage order: stated confidence ascending.
Checkpoint per variant on validation by area under the selective-accuracy curve.
**Primary: population-weighted exact accuracy at 20% of reports reviewed (tie-aware estimator v2.1), each variant minus jury
vote-fraction triage, paired.** Secondary: abstention rate, Task-A AUROC and ECE of the new scores.

## E. Multi-turn agent (≈10 GPU-h)
Re-split templates so the held-out test has ≥ 20 answerable templates (moving templates out of training), seed 20261016; the split
is written to `rl_sql/templates_agent.json`. Tools: list_tables, describe(table), run_sql(query); ≤ 6 calls; final answer or refusal.
(a) project-3 SFT model single-shot; (b) the same model as an agent, no further training; (c) agent after GRPO, outcome reward
(1 correct final answer, −0.5 answering an unanswerable question, 0 otherwise), tool-output tokens masked from the loss.
Because the re-split moves templates, (a) is retrained on the new training templates (same recipe as project 3).
Metrics: execution accuracy (held-out), correct refusals, mean tool calls, accuracy by type; bootstrap over templates.
Primary: (c) − (a) execution accuracy.

## F. Perturbation prediction (public data, 1–2 weeks)
Task, dataset and split exactly as SynthPert (arXiv 2509.25346), confirmed from the paper before any download; zero-shot vs SFT vs
SFT→GRPO (correct-direction reward) with an open 4–8B model; compared against the paper's published numbers. A separate design
addendum is committed once the paper's task definition is confirmed, before training.
