# Project 3 — Text-to-SQL cohort questions with RL: DESIGN (fixed 2026-10-15, before any training run)

**Question.** Can a 7B model trained with GRPO answer unseen kinds of cohort questions over a structured ERIN label table by writing
SQL, and refuse unanswerable ones?

**Database** (`analysis/rl_sql/erin_labels.sqlite`, structured fields only, no report text, pseudonymised IDs):
patients(patient_id, sex, birth_year — only if present in the source tables, otherwise omitted and documented),
reports(report_id, patient_id, report_date, specimen_protocol), sites(report_id, site_letter, grade, jury_vote_fraction, organ_site,
specimen_type), slides(slide_id, report_id, site_letter, stain). report_id/patient_id are re-keyed to opaque integers.

**Questions.** 30+ templates with gold SQL (counts, grade distributions, progression within N years, time to first HGD+, patients with
≥ k reports, per-year counts, worst grade per patient, site- vs case-level), instantiated to ~1,500 questions, 2 paraphrases each from
a local LLM (Qwen3-8B, ollama/vLLM), plus ~15% unanswerable/ambiguous questions whose gold behaviour is the token `CANNOT_ANSWER`.
Gold answers = execution of the gold SQL. **Split by template**: ~60% of templates train, ~15% validation, ~25% test; test templates
never seen in training.

**Models.** Base Qwen2.5-Coder-7B-Instruct, LoRA r = 16. Baselines: zero-shot with schema doc; 5-shot; SFT on gold SQL. RL: GRPO,
reward 1 if executed result equals gold, 0.1 if SQL runs but differs, 0 if it fails; unanswerable: 1 for CANNOT_ANSWER, −0.5 for any SQL.
Checkpoint chosen on validation templates.

**Primary endpoint.** Execution accuracy on held-out test templates, RL vs SFT, bootstrap over templates (2,000).
**Secondary.** zero-/few-shot; correct-refusal rate; accuracy by question type. Optional agent stage only if time allows.
