# Reading pathology reports to teach computers to grade Barrett's oesophagus

**In plain words.** People with Barrett's oesophagus have regular biopsies, and a pathologist writes a report saying whether the cells look normal, pre-cancerous (dysplasia) or cancerous. This project uses local AI language models to read thousands of those reports and turn them into labels, then uses the labels to train image models that grade the scanned biopsy slides. Along the way it measures how often the AI reading is wrong, how little human checking is needed to fix it, and how much data the image models really need.

This repository contains the code only. No patient data, report text, labels, images or model weights are stored here; see [Data](#data).

---

## Abstract

Routine pathology archives pair millions of whole-slide images with free-text reports, but the reports are unstructured and the grade of each biopsy site is buried in prose. We built a fully local pipeline on 7,101 reports and 12,343 scanned slides from 2,537 patients in a UK Barrett's oesophagus surveillance archive (ERIN). A jury of ten open-weight language models, run on cluster GPUs with no external API, reads each report and assigns a grade to every lettered biopsy site. Against 479 sections graded by hand, the jury's consensus agrees on the exact grade in 93% of sections (population-weighted) and on the dysplasia yes/no decision in 98.6%. The jury's own vote fraction is a strong error signal: sending the 20% of reports with the least agreement to a human lifts exact accuracy from 93.0% to 97.0%, about 2.9 hours of reading for this archive. When attention-based multiple-instance models are trained on these labels, per-section labels beat the common shortcut of giving every slide its case's worst grade (binary dysplasia AUROC 0.950 vs 0.936; six-class macro-AUROC 0.881 vs 0.818). Learning curves are flat above about 100 labelled patients. A zero-shot pathology vision–language model (CONCH) is far below supervised training (0.675 vs 0.950), and an image-budget study shows that 16 well-chosen tiles per slide perform as well as 1,200 random ones. The jury can be distilled: a 150-million-parameter clinical text encoder (BioClinical ModernBERT) fine-tuned on the jury's labels agrees with the human on 93.3–94.7% of sections across three seeds, level with the ten-model jury at about 0.003 s per report. A 4-billion-parameter language model fine-tuned on the same labels also reaches jury level (93–94%); adding reinforcement learning lowers it (89–91%).

## 1. Introduction

Barrett's oesophagus is the precursor of oesophageal adenocarcinoma, and its management depends on a histological grade: non-dysplastic Barrett's (NDBE), indefinite for dysplasia (IND), low-grade dysplasia (LGD), high-grade dysplasia (HGD) or cancer. The grade is hard to assign, and pathologists disagree often, especially for LGD and IND [Montgomery 2001; Kerkhof 2007; Vennalaganti 2017]. AI models for slide grading need many labelled slides [Campanella 2019; Bouzid 2024], while expert annotation is scarce and expensive.

Every routine biopsy already comes with a report that states the grade. If a machine can read those reports reliably, the archive labels itself. Earlier work used rule-based or classical NLP to pull dysplasia status from Barrett's reports [Nguyen Wenker 2023; Zecevic 2024], and recent studies show that large language models extract structured diagnoses from pathology reports with high accuracy [Truhn 2024; Grothey 2025; Johnson 2025], including Barrett's-specific categories [Kefeli 2023]. This project addresses three open questions:

1. **Section level.** A report usually covers several biopsy sites (A, B, C…) with different grades, while each slide comes from one site. Most report-derived labelling works at the case level. Does section-level labelling matter for training slide models?
2. **Knowing when the labels are wrong.** Report-derived labels are noisy, and label errors distort both training and evaluation [Northcutt 2021; Jain 2021]. Can the labeller tell us where to spend scarce human review [Raghu 2019; Bernhardt 2022]?
3. **Privacy.** NHS report text cannot leave the institution. Can open-weight models running on local GPUs do the job? Ensembles of locally deployed models have been proposed for exactly this setting [Tripathi 2025].

## 2. Data

All data stay on the Cancer Research UK Cambridge Institute cluster and are never copied into this repository.

| Dataset | What it is | Size |
|---|---|---|
| **ERIN** | Routine upper-GI pathology reports (redacted free text) with scanned H&E and special-stain slides, Cambridge, 2014–2025 | 7,101 reports, 2,537 patients, 12,343 slides (11,672 H&E) |
| ERIN oesophageal/GOJ subset | Slides whose lettered section is oesophageal or junctional, used for slide-model training | 7,924 slides, 1,603 patients |
| Hand-label sample | 485 sections in 143 reports, sampled at random plus strata by grade, jury agreement and special cases; graded blind by one rater, with inverse-probability sampling weights; 39 blind repeats | 479 graded sections, 130 patients |
| **SWG** | A research cohort with pathologist block grades linked to the clinical database [Killcoyne 2020] | 707 slides; 427 report × block pairs for external checks |

Patients shared between ERIN and SWG (70) are excluded from every evaluation that uses both. Tile features come from the UNI2-h foundation model [Chen 2024] at 0.5 µm per pixel, 224-pixel tiles (22.6 million tiles).

## 3. Methods

### 3.1 The labelling jury (`labeller/`, `campaigns/allslides/`)
Ten open-weight models (Gemma 3 12B/27B, Phi-4, Qwen3 14B/32B, DeepSeek-R1 14B, Mistral Small 3.2, Llama 3.1 8B, Granite 3.3 8B, Llama 3.3 70B) run through Ollama on the cluster. Each reads the full report and returns, for every lettered section, the organ site, specimen type, all grades present and any cancer subtype as JSON (prompt v3). A section's consensus grade keeps any grade listed by at least half of the jurors who returned that section, and the site comes from a plurality vote. The fraction of jurors agreeing with the consensus worst grade is the section's vote fraction. A keyword ladder ("pathladder") is the rule-based baseline.

### 3.2 Human reference (`analysis/erin_handlabel/`, `scripts/grading_app.py`)
A stratified sample was drawn from the population of oesophageal/GOJ sections with a scanned slide, with known inclusion probabilities, so that accuracy can be re-weighted to the whole archive. Sections were graded one per screen in a keyboard-only web app, with the jury's answer hidden. A second rater is grading the same items.

### 3.3 Juror benchmark (`analysis/juror_benchmark/`)
Every juror, a top-3 subset chosen on a patient-split half, the full jury and the rule baseline are compared with the human grades on the random slice, the re-weighted sample and the held-out half.

### 3.4 Slide models (`analysis/isbi_rerun/`, `scripts/abmil_*.py`)
Gated-attention multiple-instance learning [Ilse 2018] over UNI2-h tile features, trained with three label sources: the case's worst grade (arm A), the slide's own section grade (arm B), and B restricted to sections with vote fraction ≥ 0.75 (arm C). The splits are patient-disjoint 5-fold, the 130 hand-labelled patients are held out entirely, three seeds are used, and the whole feature set sits on the GPU so that each fold job trains 36 models. Test sets: T1 (held-out folds, jury truth), T2 (human truth). Confidence intervals come from a patient-clustered bootstrap with 2,000 resamples.

### 3.5 Triage, auditing and zero-shot baselines (`analysis/paperB_extras/`)
*Triage*: rank sections by a confidence signal, assume reviewed sections become correct, and estimate population accuracy against review budget with a Horvitz–Thompson estimator on the weighted human sample. *Auditor*: does disagreement between the image model and the text label flag text errors? *Zero-shot*: CONCH [Lu 2024] with a fixed prompt ensemble, scored on exactly the supervised test slides, with tile scores pooled by mean, max or top-k. *Label scale*: arm B retrained on 25 to 1,000 patients per fold.

### 3.6 Tile-budget study (`analysis/tilebudget/`)
A pre-registered study of how many tiles, and which ones, a slide model needs. Training budgets range from 4 tiles to all tiles. Test budgets range from 1 tile to all. Selection rules: random, spatial grid, attention-top, attention-bottom, mixed, CONCH-ranked and feature-diverse. Aggregators: attention MIL, mean pooling with a linear head, and max.

### 3.7 Single-model graders: distillation and reinforcement learning (`scripts/rl/`)
**Distilling the jury** (`scripts/rl/explore/clf/`). BioClinical ModernBERT-base [Sounack 2025], a 150M-parameter long-context clinical encoder built on ModernBERT [Warner 2024], is fine-tuned as a six-way classifier on the same jury-labelled training sections as the reinforcement-learning models (section letter prepended to the report text; checkpoint chosen on validation agreement with the jury; three seeds), with a TF-IDF + logistic-regression baseline.

**Reinforcement learning.** Three pre-registered projects ([`scripts/rl/DESIGN_p1.md`](scripts/rl/DESIGN_p1.md)–[`p3`](scripts/rl/DESIGN_p3.md)), with the 479 human-labelled sections kept strictly for testing and every hand-labelled patient removed from training:
(1) one 4B model (MedGemma-4B [Sellergren 2025]) trained with GRPO [Shao 2024] against the jury's high-confidence labels, compared with zero-shot and supervised fine-tuning (SFT), following the Text2SQL RL recipe of Stoisser et al. [2025];
(2) the same model rewarded for answering UNSURE, compared with ten-model disagreement at a matched review budget;
(3) a 7B model (Qwen2.5-Coder-7B) that answers cohort questions by writing SQL over a pseudonymised label table, with query templates held out for testing.
Pre-registered follow-ups ([`scripts/rl/DESIGN_followups.md`](scripts/rl/DESIGN_followups.md)) measure calibration directly (error-detection AUROC, expected calibration error), repeat project 1 with two more seeds, try two alternative uncertainty rewards (an UNSURE reward conditioned on jury disagreement; a Brier-scored stated confidence), turn project 3 into a multi-turn tool-using agent (`list_tables`, `describe`, `run_sql`; at most 6 calls) trained with GRPO, and apply the same SFT→GRPO recipe to public gene-perturbation data (PerturbQA [Wu 2025], as used by SynthPert [2025]).
Exploratory strands (`scripts/rl/explore/`) test temperature and isotonic recalibration and conformal prediction sets (`calib/`), a logistic probe on the model's hidden state (`probe/`), seed-vote ensembles and a jury-vote-share reward (`grader/`), expert iteration with a refusal penalty for the SQL agent (`agent/`), and knowledge-graph nearest-neighbour retrieval for PerturbQA (`pert/`).

### 3.8 Compute (`scripts/cluster/`, `feasibility/run_task.sh`)
Every job is small and idempotent. Each one is submitted twice, to two GPU partitions, takes a lock when it starts and cancels its pending twin. Pull workers hold an allocation and drain the task list. Report text and slides are never read inside GPU jobs: tiles are cut once on CPU nodes and stored.

## 4. Results

| Question | Result |
|---|---|
| How good is the jury? | 93.0% exact grade and 98.6% dysplasia yes/no agreement with the human, population-weighted. On the random slice of 50 sections: 94.0% exact. Repeat-labelling agreement of the human: 39 of 39. |
| Best single juror? | No single juror beats the jury on the unbiased sample. On the held-out half, Mistral Small 3.2 is best (0.90 exact). A top-3 subset buys nothing over the full jury. The keyword rule reaches 0.64–0.75. |
| Is vote fraction an error signal? | Exact agreement is 0.96 when all jurors agree, 0.92 at 0.75–1, 0.51 at 0.5–0.75 and 0.31 at 0.5 or below. |
| How much human review is needed? | Reviewing the 20% of reports with the lowest vote fraction lifts exact accuracy from 93.0% to 97.0% (≈2.9 h at the median 23.6 s per report; ≈7.5 h at the mean). A learned combination of signals reaches 97% at 18% of reports, not significantly better than plain vote fraction. |
| Can the image model catch text errors? | Partly. In 54% of the six-class jury errors the image model agrees with the human, but in only 26% of the dysplasia yes/no errors. Adding image disagreement to vote fraction raises error-detection AUROC from 0.806 to 0.856, with a CI spanning zero. |
| Section labels or case labels? | Section labels win. Binary LGD+ AUROC 0.950 vs 0.936 (Δ +0.014 [+0.007, +0.021]). Six-class macro-AUROC 0.881 vs 0.818 (Δ +0.063 [+0.041, +0.084]). Biggest gain for LGD (+0.17). The case label disagrees with the section label for 19.5% of slides. |
| How many labelled patients? | Supervised training matches the full set from about 100 training patients (150 for six-class), and beats zero-shot from 25–50 patients. |
| Zero-shot pathology VLM? | CONCH reaches 0.675 binary with mean pooling and 0.827 with the single most suspicious tile, against 0.950 supervised. It is near chance for HGD and LGD. An earlier 0.889 came from a report-supervised model trained on ERIN itself, not a zero-shot one. |
| How many tiles per slide? | Random tiles: 0.858 with 1 tile, 0.932 with 16, 0.948 with 256, 0.950 with 1,200, 0.952 with all. Sixteen attention-ranked tiles: 0.945. The "most normal" tiles carry almost no signal (0.686 with 1 tile). Grid or diversity sampling is worse than random at small budgets. Training on 256 random tiles costs nothing measurable. |
| Can a small model replace the jury? | Yes, by distillation. BioClinical ModernBERT fine-tuned on the jury's labels agrees with the human on 93.3%, 94.5% and 94.7% of sections (three seeds), against 93.0% for the jury (+0.3 to +1.8, CIs include zero) and 89.1% for the reinforcement-learning model (+4.2 to +5.6, CIs exclude zero). Dysplasia yes/no 98.2–98.3%. It runs in 0.003 s per report, 27 times smaller and 15 times faster than the 4B model. TF-IDF + logistic regression reaches 72.8%. Ranked by its own softmax probability, it reaches 97.4–98.2% after review of 20% of reports. |
| Can a fine-tuned language model replace the jury? (RL project 1) | Yes, with supervised fine-tuning on the answer only. MedGemma-4B fine-tuned with the loss on the grade alone agrees with the human on 93.0%, 94.4% and 93.4% of sections (three seeds; see correction), level with the jury. Adding GRPO reinforcement learning on top lowers it to 89.1–90.6% (−3.8 [−6.0, −2.0]). MedGemma-27B zero-shot reaches 87.3%. |
| Can one model know when it is unsure? (RL project 2) | Not through abstention: rewarded for answering UNSURE, the model almost never did (under 0.1% of sections). Its grade-token probability is a weaker triage signal than jury disagreement once sections from training patients are excluded (−3.0 [−5.9, −0.9] at 20% of reports). |
| Does reinforcement learning help? (3 seeds) | No. An earlier comparison reported RL adding +6.6 points over SFT, but that SFT baseline computed its loss over the whole report rather than the answer. Against SFT with the loss on the answer only, RL is −3.9, −3.8 and −3.7 points in three seeds (mean −3.8 [−6.0, −2.0]); majority voting over 8 SFT samples adds nothing. |
| Does the model know its own errors? | The jury's vote fraction flags its errors with AUROC 0.86 and is close to calibrated (ECE 0.03). The RL model's grade-token probability ranks errors as well (AUROC 0.91; Δ +0.05 [−0.05, +0.15]) but is over-confident (ECE 0.10). Eight-sample agreement is useless (0.58). |
| Can a different reward teach abstention? | No. With UNSURE rewarded on jury-disputed sections the model abstained on none of the test sections; with a Brier-scored stated confidence it said 100 on every section. |
| Does RL help text-to-SQL cohort questions? (RL project 3) | No. On held-out query templates GRPO answers 47.5% correctly against 52.5% for SFT (−5.0 [−22.6, +12.4]) and refuses 57% of unanswerable questions against 98% for SFT. |
| Does a tool-using agent help? | RL turns a poor agent into a usable one (held-out accuracy 20% → 59%, correct refusals 9% → 58%), but it does not beat single-shot SFT (+12.1 [−20.7, +52.7] by question; +1.7 [−10.0, +15.1] averaged over templates). |
| Does the recipe transfer to gene-perturbation prediction? | No gain from RL. On PerturbQA, SFT lifts direction-of-change AUROC from 0.50 to 0.62; GRPO adds −0.002 [−0.011, +0.007]. Detecting a change stays at 0.57, level with the published SFT baseline (0.59–0.60) and below SynthPert (0.70–0.79). |
| Why does RL stall? | Once the model is mostly right, the eight sampled answers in a GRPO group are identical and give no gradient: 78% of groups early in training, 94% by the end, and 95–97% in the abstention runs. Voting the three RL seeds with MedGemma-27B and a short vote-share-reward run reaches 91.2% (+2.1 [+0.5, +4.2] over one seed), still short of the jury. |
| Can single-model confidence be fixed? | Partly. Temperature scaling on validation cuts the RL model's calibration error from 0.098 to 0.069 (0.007 against the jury's own labels). A probe on the hidden state finds errors no better than the token probability (AUROC 0.92 vs 0.91). At 10% of reports nothing beats jury vote fraction (95.7%). |
| Can the SQL agent beat single-shot SQL? | Narrowly. Expert iteration plus a refusal penalty reaches 46.0% averaged over held-out templates against 40.9% for single-shot SFT (+5.1 [+1.0, +10.2]). False refusals fall from 23% to 0%, but many become failed queries against the wrong table. |
| What drives perturbation-prediction AUROC? | Retrieval. With no language model, averaging the outcomes of the 20 training knockdowns most similar to the query (knowledge-graph neighbours combined with gene-description text) reaches DE 0.645 / DIR 0.734 on PerturbQA, against 0.566 / 0.617 for the fine-tuned LLM. An LLM given those retrieved outcomes in its prompt reproduces them (0.630 / 0.706); adding gene descriptions lowers it. |
| Does the archive transfer to the research cohort? | Weakly. ERIN-trained models reach 0.571 on SWG and SWG-trained models 0.684 on ERIN. SWG positives are LGD only. |

Full tables, confidence intervals and deviations from each design are in the result files on the cluster; the code that produces each number is in the folder named in the Methods.

## Corrections (26 Oct 2026)

- Supervised fine-tuning baselines in RL projects 1–2 and the seed follow-up computed the training loss over the whole conversation (report text included), not only the answer. With the loss on the answer only, fine-tuning alone reaches 93.0–94.4% and reinforcement learning on top is 3.8 points worse. Earlier versions of this page said reinforcement learning added 6.6 points.
- Single-model triage was scored on a population that included 2,999 sections from training patients. Excluding them, the RL model's token probability is 3.0 points worse than jury triage at 20% of reports (earlier: level); a cross-fitted encoder ties the jury (+0.1 [−0.4, +0.6]).

## 5. Discussion

Local open-weight models can read routine pathology reports well enough to label an archive at the level of the biopsy site, and where they disagree is where they tend to be wrong. The labels matter for what is trained on them: assigning each slide its case's worst grade, still common in report-supervised slide work [Campanella 2019; Bouzid 2024], costs real accuracy, most of all for low-grade dysplasia, the hardest and most consequential class. Once the labels are at the right granularity, the learning curves flatten at about 100 patients and 16 well-chosen tiles do the work of 1,200, so further gains are more likely to come from better labels and better tile choice than from more slides or more compute.

The single-model results point the same way. A small model trained on the jury's labels, whether a 150M encoder or a fine-tuned 4B language model, matches the jury, so the expensive ensemble is needed once, to label the archive, and not at inference; one strong juror's labels teach nearly as well as the full jury's. The encoder's own confidence triages as well as jury disagreement when every score is out-of-fold. Reinforcement learning on top of fine-tuning did not help here: an apparent +6.6-point gain came from a fine-tuning baseline whose loss covered the whole report. For gene-perturbation prediction, most of the measurable signal comes from per-knockdown and per-gene shortcuts that simple retrieval captures; under a shift-corrected metric fine-tuned language models are at chance for detecting a change.

**Limitations.** There is a single, non-pathologist human rater (a second rater is in progress). Reviewed labels are assumed correct. The data come from one centre. The image models are trained on jury labels, so they share the jury's biases. Sampling weights assume independent draws. Every number here is internal validation except the SWG checks.

## 6. Repository layout

```
labeller/            report-labelling jury (whole-report and per-section), rules, audits
campaigns/allslides/ all-slide tile extraction workers and the per-section jury v3
scripts/             slide models, grading app, thesis analyses, cluster tooling (cluster/), RL projects (rl/)
analysis/<study>/    code for each study: isbi_rerun, juror_benchmark, erin_handlabel, erin_swg_linkage, paperB_extras, tilebudget
feasibility/         race-to-run task runner and early feasibility tasks
tools/publish/       how this public mirror is made (see below)
```

Paths inside the scripts point to the authors' cluster. The code is published for transparency and reuse, not as a turnkey package.

**How this mirror is kept safe.** `tools/publish/publish.sh` copies only code files (no CSV, JSON, images or weights; nothing over 1 MB) from the cluster and refuses to publish if any file contains a real report number, a patient pseudonym, a secret or token, an e-mail address, or any 8-word run of text that appears in fewer than ten real reports.

## References

- Bernhardt M et al. Active label cleaning for improved dataset quality under resource constraints. *Nature Communications* 2022. doi:10.1038/s41467-022-28818-3
- Bouzid K et al. Enabling large-scale screening of Barrett's esophagus using weakly supervised deep learning in histopathology. *Nature Communications* 2024. doi:10.1038/s41467-024-46174-2
- Campanella G et al. Clinical-grade computational pathology using weakly supervised deep learning on whole slide images. *Nature Medicine* 2019. doi:10.1038/s41591-019-0508-1
- Chen RJ et al. Towards a general-purpose foundation model for computational pathology. *Nature Medicine* 2024. doi:10.1038/s41591-024-02857-3
- Gehrung M et al. Triage-driven diagnosis of Barrett's esophagus for early detection of esophageal adenocarcinoma using deep learning. *Nature Medicine* 2021. doi:10.1038/s41591-021-01287-9
- Grothey B et al. Comprehensive testing of large language models for extraction of structured data in pathology. *Communications Medicine* 2025. doi:10.1038/s43856-025-00808-8
- Ilse M, Tomczak JM, Welling M. Attention-based deep multiple instance learning. *ICML* 2018. arXiv:1802.04712
- Jain S et al. Effect of radiology report labeler quality on deep learning models for chest X-ray interpretation. *NeurIPS Workshop* 2021. arXiv:2104.00793
- Johnson et al. Large language models for extracting histopathologic diagnoses of colorectal cancer and dysplasia. *BMJ Open Gastroenterology* 2025. doi:10.1136/bmjgast-2025-001896
- Kefeli J et al. Large language models for granularized Barrett's esophagus diagnosis classification. arXiv 2023. doi:10.48550/arXiv.2308.08660
- Kerkhof M et al. Grading of dysplasia in Barrett's oesophagus: substantial interobserver variation between general and gastrointestinal pathologists. *Histopathology* 2007. doi:10.1111/j.1365-2559.2007.02706.x
- Killcoyne S et al. Genomic copy number predicts esophageal cancer years before transformation. *Nature Medicine* 2020. doi:10.1038/s41591-020-1033-y
- Lu MY et al. A visual-language foundation model for computational pathology. *Nature Medicine* 2024. doi:10.1038/s41591-024-02856-4
- Montgomery E et al. Reproducibility of the diagnosis of dysplasia in Barrett esophagus: a reaffirmation. *Human Pathology* 2001. doi:10.1053/hupa.2001.23510
- Nguyen Wenker T et al. Using natural language processing to automatically identify dysplasia in pathology reports for patients with Barrett's esophagus. *Clinical Gastroenterology and Hepatology* 2023. doi:10.1016/j.cgh.2022.09.005
- Northcutt CG, Athalye A, Mueller J. Pervasive label errors in test sets destabilize machine learning benchmarks. *NeurIPS Datasets and Benchmarks* 2021. arXiv:2103.14749
- Raghu M et al. The algorithmic automation problem: prediction, triage, and human effort. arXiv 2019. doi:10.48550/arXiv.1903.12220
- Sellergren A et al. MedGemma technical report. arXiv 2025. arXiv:2507.05201
- Shao Z et al. DeepSeekMath: pushing the limits of mathematical reasoning in open language models (introduces GRPO). arXiv 2024. arXiv:2402.03300
- Sounack T, Davis J, et al. BioClinical ModernBERT: a state-of-the-art long-context encoder for biomedical and clinical NLP. arXiv 2025. arXiv:2506.10896
- Stoisser J, Boubnovski Martell M, Fauqueur J. Sparks of tabular reasoning via Text2SQL reinforcement learning. arXiv 2025. arXiv:2505.00016
- SynthPert. arXiv 2025. arXiv:2509.25346
- Warner B, et al. Smarter, better, faster, longer: a modern bidirectional encoder for fast, memory efficient, and long context finetuning and inference (ModernBERT). arXiv 2024. arXiv:2412.13663
- Wu M, et al. PerturbQA benchmark. ICLR 2025. github.com/genentech/PerturbQA
- Tripathi A et al. Employing consensus-based reasoning with locally deployed LLMs for enabling structured data extraction from surgical pathology reports. *medRxiv* 2025. doi:10.1101/2025.04.22.25326217
- Truhn D et al. Extracting structured information from unstructured histopathology reports using generative pre-trained transformer 4. *Journal of Pathology* 2024. doi:10.1002/path.6232
- Vennalaganti P et al. Discordance among pathologists in the United States and Europe in diagnosis of low-grade dysplasia for patients with Barrett's esophagus. *Gastroenterology* 2017. doi:10.1053/j.gastro.2016.10.041
- Zecevic et al. Automated decision making in Barrett's oesophagus: development and deployment of a natural language processing tool. *npj Digital Medicine* 2024. doi:10.1038/s41746-024-01302-6

## Contact

Rehan Zuberi, Markowetz lab, Cancer Research UK Cambridge Institute, University of Cambridge.
