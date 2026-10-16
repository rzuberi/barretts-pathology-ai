# Reading pathology reports to teach computers to grade Barrett's oesophagus

**In plain words.** People with Barrett's oesophagus have regular biopsies, and a pathologist writes a report saying whether the cells look normal, pre-cancerous (dysplasia) or cancerous. This project uses local AI language models to read thousands of those reports and turn them into labels, then uses the labels to train image models that grade the scanned biopsy slides. Along the way it measures how often the AI reading is wrong, how little human checking is needed to fix it, and how much data the image models really need.

This repository contains the code only. No patient data, report text, labels, images or model weights are stored here; see [Data](#data).

---

## Abstract

Routine pathology archives pair millions of whole-slide images with free-text reports, but the reports are unstructured and the grade of each biopsy site is buried in prose. We built a fully local pipeline on 7,101 reports and 12,343 scanned slides from 2,537 patients in a UK Barrett's oesophagus surveillance archive (ERIN). A jury of ten open-weight language models, run on cluster GPUs with no external API, reads each report and assigns a grade to every lettered biopsy site. Against 479 sections graded by hand, the jury's consensus agrees on the exact grade in 93% of sections (population-weighted) and on the dysplasia yes/no decision in 98.6%. The jury's own vote fraction is a strong error signal: sending the 17% of reports with the least agreement to a human lifts exact accuracy from 93.0% to 97.0%, about 2.4 hours of reading for this archive. When attention-based multiple-instance models are trained on these labels, per-section labels beat the common shortcut of giving every slide its case's worst grade (binary dysplasia AUROC 0.950 vs 0.936; six-class macro-AUROC 0.881 vs 0.818). Learning curves are flat above about 100 labelled patients. A zero-shot pathology vision–language model (CONCH) is far below supervised training (0.675 vs 0.950), and an image-budget study shows that 16 well-chosen tiles per slide perform as well as 1,200 random ones. A single 4-billion-parameter model trained with supervised fine-tuning and then reinforcement learning reaches 89.1% exact agreement, 3.9 points behind the ten-model jury at a small fraction of its cost.

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

### 3.7 Reinforcement learning (`scripts/rl/`)
Three pre-registered projects ([`scripts/rl/DESIGN_p1.md`](scripts/rl/DESIGN_p1.md)–[`p3`](scripts/rl/DESIGN_p3.md)), with the 479 human-labelled sections kept strictly for testing and every hand-labelled patient removed from training:
(1) one 4B model (MedGemma-4B [Sellergren 2025]) trained with GRPO [Shao 2024] against the jury's high-confidence labels, compared with zero-shot and supervised fine-tuning (SFT), following the Text2SQL RL recipe of Stoisser et al. [2025];
(2) the same model rewarded for answering UNSURE, compared with ten-model disagreement at a matched review budget (running);
(3) a 7B model (Qwen2.5-Coder-7B) that answers cohort questions by writing SQL over a pseudonymised label table, with query templates held out for testing.

### 3.8 Compute (`scripts/cluster/`, `feasibility/run_task.sh`)
Every job is small and idempotent. Each one is submitted twice, to two GPU partitions, takes a lock when it starts and cancels its pending twin. Pull workers hold an allocation and drain the task list. Report text and slides are never read inside GPU jobs: tiles are cut once on CPU nodes and stored.

## 4. Results

| Question | Result |
|---|---|
| How good is the jury? | 93.0% exact grade and 98.6% dysplasia yes/no agreement with the human, population-weighted. On the random slice of 50 sections: 94.0% exact. Repeat-labelling agreement of the human: 39 of 39. |
| Best single juror? | No single juror beats the jury on the unbiased sample. On the held-out half, Mistral Small 3.2 is best (0.90 exact). A top-3 subset buys nothing over the full jury. The keyword rule reaches 0.64–0.75. |
| Is vote fraction an error signal? | Exact agreement is 0.96 when all jurors agree, 0.92 at 0.75–1, 0.51 at 0.5–0.75 and 0.31 at 0.5 or below. |
| How much human review is needed? | Reviewing the 17% of reports with the lowest vote fraction lifts exact accuracy from 93.0% to 97.0% (≈2.4 h at the median 23.6 s per report; ≈6.4 h at the mean). A learned combination of signals does not beat plain vote fraction. |
| Can the image model catch text errors? | Partly. In 54% of the six-class jury errors the image model agrees with the human, but in only 26% of the dysplasia yes/no errors. Adding image disagreement to vote fraction raises error-detection AUROC from 0.806 to 0.856, with a CI spanning zero. |
| Section labels or case labels? | Section labels win. Binary LGD+ AUROC 0.950 vs 0.936 (Δ +0.014 [+0.007, +0.021]). Six-class macro-AUROC 0.881 vs 0.818 (Δ +0.063 [+0.041, +0.084]). Biggest gain for LGD (+0.17). The case label disagrees with the section label for 19.5% of slides. |
| How many labelled patients? | Supervised training matches the full set from about 100 training patients (150 for six-class), and beats zero-shot from 25–50 patients. |
| Zero-shot pathology VLM? | CONCH reaches 0.675 binary with mean pooling and 0.827 with the single most suspicious tile, against 0.950 supervised. It is near chance for HGD and LGD. An earlier 0.889 came from a report-supervised model trained on ERIN itself, not a zero-shot one. |
| How many tiles per slide? | Random tiles: 0.858 with 1 tile, 0.932 with 16, 0.948 with 256, 0.950 with 1,200, 0.952 with all. Sixteen attention-ranked tiles: 0.945. The "most normal" tiles carry almost no signal (0.686 with 1 tile). Grid or diversity sampling is worse than random at small budgets. Training on 256 random tiles costs nothing measurable. |
| Can one small model replace the jury? (RL project 1) | Nearly. MedGemma-4B after SFT then GRPO agrees with the human on 89.1% [83.4, 93.6] of sections, against 42.9% zero-shot, 82.3% after SFT alone (RL adds +6.9 [+2.7, +11.8] over SFT) and 87.3% for MedGemma-27B zero-shot. It stays 3.9 points [0.0, 7.5] behind the ten-model jury while running in about 0.05 s per report. |
| Does RL help text-to-SQL cohort questions? (RL project 3) | No. On held-out query templates GRPO answers 47.5% correctly against 52.5% for SFT (−5.0 [−22.6, +12.4]) and refuses 57% of unanswerable questions against 98% for SFT. |
| Does the archive transfer to the research cohort? | Weakly. ERIN-trained models reach 0.571 on SWG and SWG-trained models 0.684 on ERIN. SWG positives are LGD only. |

Full tables, confidence intervals and deviations from each design are in the result files on the cluster; the code that produces each number is in the folder named in the Methods.

## 5. Discussion

Local open-weight models can read routine pathology reports well enough to label an archive at the level of the biopsy site, and where they disagree is where they tend to be wrong. The labels matter for what is trained on them: assigning each slide its case's worst grade, still common in report-supervised slide work [Campanella 2019; Bouzid 2024], costs real accuracy, most of all for low-grade dysplasia, the hardest and most consequential class. Once the labels are at the right granularity, the learning curves flatten at about 100 patients and 16 well-chosen tiles do the work of 1,200, so further gains are more likely to come from better labels and better tile choice than from more slides or more compute.

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
- Stoisser J, Boubnovski Martell M, Fauqueur J. Sparks of tabular reasoning via Text2SQL reinforcement learning. arXiv 2025. arXiv:2505.00016
- Tripathi A et al. Employing consensus-based reasoning with locally deployed LLMs for enabling structured data extraction from surgical pathology reports. *medRxiv* 2025. doi:10.1101/2025.04.22.25326217
- Truhn D et al. Extracting structured information from unstructured histopathology reports using generative pre-trained transformer 4. *Journal of Pathology* 2024. doi:10.1002/path.6232
- Vennalaganti P et al. Discordance among pathologists in the United States and Europe in diagnosis of low-grade dysplasia for patients with Barrett's esophagus. *Gastroenterology* 2017. doi:10.1053/j.gastro.2016.10.041
- Zecevic et al. Automated decision making in Barrett's oesophagus: development and deployment of a natural language processing tool. *npj Digital Medicine* 2024. doi:10.1038/s41746-024-01302-6

## Contact

Rehan Zuberi, Markowetz lab, Cancer Research UK Cambridge Institute, University of Cambridge.
