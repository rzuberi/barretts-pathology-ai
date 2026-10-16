# Project 1 — RL-trained single-model section grader: DESIGN (fixed 2026-10-15, before any training run)

**Question.** Can one 4B model, trained with GRPO against the v3 jury's high-confidence section labels, match the ten-model jury
on Rehan's human labels?

**Data** (`analysis/rl_grader/data/`, built by `scripts/rl/build_data.py`; counts in `data/summary.json`).
Item = one ERIN report (gross + final diagnosis + microscopic, first 8,000 characters) + one section letter (A–J or WHOLE).
Label = v3 jury consensus worst grade of that section (NORMAL_OTHER, NDBE, IND, LGD, HGD, CANCER).
Excluded from all training/selection data: every patient in the hand-label sample (130) and the SWG-shared patients (70); 200 total.
Remaining patients split 80/10/10 by patient (seed 0): train 14,380 / val 1,771 / jury-test 1,808 sections with jury vote fraction ≥ 0.75.
Human test set: the 479 graded sections of the 485-section hand-label sample — **test only**, never used for training, tuning,
checkpoint selection or thresholds.

**Models** (all LoRA r = 16, α = 32, dropout 0.05, on all attention+MLP projections; same budget everywhere).
- Base: `google/medgemma-4b-it` (HF access confirmed 2026-10-15).
- A. zero-shot base (prompt in `scripts/rl/common.py`: the v3 jury's grade definitions, asked about one section; JSON `{"reasoning", "grade"}`).
- B. SFT: 1 epoch over all 14,380 training sections, target `{"grade": X}`, lr 1e-4, effective batch 16.
- RL-base: GRPO from the base. RL-SFT: GRPO from the SFT adapter (merged).
  GRPO training prompts: a class-capped sample of the training split (≤ 1,200 per class; seed 0) — rare grades carry the learning signal.
  Group size 8, temperature 1.0, max completion 128 tokens, KL β = 0.04, lr 2e-5, 16 prompts per step, 300 steps (one pass over
  ~4,800 prompts), checkpoint every 50 steps.
  Reward: +1 exact grade; +0.3 wrong grade on the correct side of the dysplasia line (NORMAL_OTHER/NDBE/IND vs LGD/HGD/CANCER); 0 otherwise;
  −0.5 if the output does not parse as JSON with a valid grade.
- Checkpoint selection: for each RL run, the checkpoint with the best exact accuracy on the validation split (greedy decoding).
  "The RL model" for the primary endpoint = the better of RL-base and RL-SFT on validation exact accuracy.
- References on the same sections: the full v3 jury (its consensus worst grade) and MedGemma-27B zero-shot (ollama, same single-section prompt).

**Primary endpoint.** Population-weighted exact worst-grade agreement with Rehan's labels on the 479 sections
(weights = the hand-label sampling weights used in Paper B triage), RL model minus zero-shot base, paired patient-clustered bootstrap
(2,000). Success threshold for a CV claim: RL within 2 points of the jury's 93.0%.

**Secondary.** RL vs SFT; RL vs jury; dysplasia yes/no agreement (LGD+ vs not); JSON parse rate; per-class confusion with IND
over-calls; agreement on the jury-test fold; unweighted and n = 50 random-slice numbers; seconds per report vs the jury.
Agreement with clinical-record grades (n = 427 DB report × block set) is attempted only if the DB report text can be fed through the same
prompt without new assumptions; otherwise reported as not done.

**Stopping rule.** Fixed 300 GRPO steps per run, no early stopping on any test set; a run whose validation exact accuracy is below the
zero-shot base at every checkpoint is reported as a null result, not retried with new hyperparameters.

**Compute.** Smoke first: 20 GRPO steps on 50 examples incl. evaluation. Race-to-run twins on cuda + h200, resume from the last checkpoint.
