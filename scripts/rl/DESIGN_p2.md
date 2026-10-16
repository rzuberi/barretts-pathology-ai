# Project 2 — Rewarding abstention: DESIGN (fixed 2026-10-15, before any training run)

**Question.** Can a single model, rewarded for saying UNSURE, flag the sections it would get wrong as well as the ten-model jury's
disagreement does?

**Data.** As project 1, but training uses *all* training-patient sections, including the 1,028 with jury vote fraction < 0.75; label =
jury consensus. Same exclusions, same validation and test splits; 479 human-labelled sections test only.

**Model.** Start from the project-1 RL model (the validation-selected one). Prompt = project 1 plus "If you cannot tell the grade
reliably, answer UNSURE." GRPO, same hyperparameters as project 1 except reward: correct grade +1, UNSURE +α, wrong grade −1,
unparseable −1. α ∈ {0.2, 0.4, 0.6}, one run each, 300 steps, checkpoint chosen on validation by the area under the selective-accuracy
curve (no human labels).

**Baselines (same 479 sections).** (a) jury vote-fraction triage (existing Paper B v2 result); (b) project-1 model confidence:
mean token probability of the grade token, and agreement across 8 samples at temperature 1.0; (c) random abstention.

**Primary endpoint.** Population-weighted exact accuracy on the 479 sections at a matched review budget of 17% of reports opened,
abstained sections counted as sent to a human (correct), exactly as in the Paper B triage estimator v2. The RL-abstention model's
operating point is its own abstention rate; for the curve, its sections are ordered by (abstain first, then grade-token probability).
Comparison: best α vs jury vote-fraction triage at 17% of reports; paired patient bootstrap (2,000).

**Secondary.** Selective accuracy and coverage per α; risk–coverage curves for all methods; overlap of abstentions with the 186
sections Rehan marked ambiguous; cost (one model vs ten).

**Stopping rule.** Three α runs, fixed steps; no re-tuning after looking at the human set.
