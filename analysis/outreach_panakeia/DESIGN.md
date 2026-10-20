# Can UNI2-h + ABMIL predict chromosomal instability from H&E in gastro-oesophageal adenocarcinoma? (TCGA only)

Pre-registration. Written 2026-10-19, before any feature, label join or model output for this study existed.
Exploratory mini-project (budget ~1 GPU-day, ~2 calendar days), not a paper. The predictions and thresholds below
are frozen; anything learned later goes in the **Status log** at the bottom, never back into the sections above.

## Data (public only)
- **Slides:** TCGA diagnostic FFPE slides (GDC open access, `experimental_strategy = Diagnostic Slide`), projects
  TCGA-STAD and TCGA-ESCA. One slide per patient: the lowest `DX` number (DX1 when present). Slides already on the
  cluster (`/mnt/scratche/fast/fmlab/datasets/imaging/tcga_{stad,esca}/slides`, downloaded from the GDC in Aug 2026)
  are reused after an MD5 check against the GDC manifest; missing ones are downloaded per slide via the GDC API with
  size + MD5 checks.
- **Labels:** cBioPortal `stad_tcga_pan_can_atlas_2018`, `esca_tcga_pan_can_atlas_2018` (patient + sample clinical
  data via the public API) and the PanCanAtlas ABSOLUTE table `TCGA_mastercalls.abs_tables_JSedit.fixed.txt`
  (GDC file 4f277128-f793-4354-a13d-30cc7fe9f6b5). Raw downloads are kept in `labels_raw/`.
- **Never used:** ERIN, OCCAMS, SWG or any NHS / controlled-access data, labels or models trained on them. The thesis
  repo has earlier TCGA analyses that pooled TCGA with OCCAMS (WGD "visibility" / transfer scripts); this study
  re-derives everything from the public sources above and does not read those results or models. It is therefore not
  blind to the fact that such analyses exist, which is stated here.

## Cohort
- TCGA-STAD (all adenocarcinoma subtypes) + TCGA-ESCA restricted to `CANCER_TYPE_DETAILED = Esophageal Adenocarcinoma`
  (squamous dropped). Expected from a GDC/cBioPortal count on 2026-10-19 (before any modelling): 413 STAD and 65 ESCA
  adenocarcinoma patients with a diagnostic slide.
- Slides that fail tile extraction (no tissue, unreadable) are excluded and counted in the flow diagram.

## Labels
| Task | Definition | Cohort |
|---|---|---|
| **WGD (primary)** | ABSOLUTE `Genome doublings` >= 1 vs 0; only `call status = called` | STAD + ESCA-adeno pooled |
| Aneuploidy (secondary, continuous) | cBioPortal `ANEUPLOIDY_SCORE` (number of arm-level events) | pooled |
| CIN subtype (secondary) | PanCanAtlas `SUBTYPE` in {STAD_CIN, ESCA_CIN} vs any other non-missing subtype (EBV, GS, MSI, POLE) | pooled |
| MSI (positive control) | `MSI_SCORE_MANTIS` >= 0.4 (MSI-H) vs < 0.4 (MSS) | STAD only |
| Cross-cohort WGD (extra) | WGD as above; train on all STAD, test on ESCA-adeno | STAD -> ESCA |

## Features
UNI2-h (local weights `phd/UNI2-h/pytorch_model.bin`, timm ViT-g/14, CLS token, 1536-d), 224 px tiles at 20x
(0.5 µm/px; 448 px at level 0 resized to 224 when the slide is scanned at 40x), tissue only (thumbnail
brightness/contrast mask, the same rule as `campaigns/extract_one.py`), at most 2,000 tissue tiles per slide drawn uniformly
at random with a fixed seed (the tile-budget study found 1,200 random tiles within 0.003 AUROC of all tiles).
Two-phase: CPU units read tiles into a uint8 pack per slide; GPU units only embed.

**Deviation from the brief, with reason:** the brief allowed reusing existing extraction. The UNI2-h features already on
the cluster for these slides (`features/20x_224px/features_uni_v2`, made by `campaigns/extract_one.py`) are **not at
20x**: that script picks the pyramid level closest to 0.5 µm/px, which on most 40x TCGA scans is level 0 (0.25 µm/px).
A check on 2026-10-19 found 407/446 files at ~0.25 µm/px and 39 at ~0.5. Magnification would then vary with scanner and
site, which is a confound for exactly this kind of study, so all slides are re-embedded at a fixed 0.5 µm/px.

## Models (same training recipe for all tasks)
- **ABMIL (main):** gated-attention ABMIL copied from `analysis/tilebudget/code/train_eval_unit.py` (d=384, attention
  128, dropout 0.25), Adam lr 1e-4, weight decay 1e-5, 25 epochs, minibatch 32 bags, 800 tiles resampled per bag per
  step, class-balanced BCE (pos_weight); MSE on z-scored target for aneuploidy. Prediction uses all tiles of the slide.
- **Mean-pool baseline:** logistic regression (ridge for aneuploidy) on the z-scored mean tile embedding (same head and
  optimiser as `meanlr_train` in the tile-budget code).
- **Label-shuffled ABMIL (null):** identical to ABMIL, training labels permuted within each training fold.
- **Site-only reference (added for interpretability, not a model of H&E):** logistic regression on one-hot TCGA tissue
  source site (+ cohort indicator), same folds. It shows how much of a label is predictable from where the slide came from.

No hyperparameter tuning: the recipe is fixed above, so there is no inner validation loop.

## Evaluation
- Patient-level 5-fold CV, stratified by label x cohort, 3 repeats (different split and model seed per repeat).
- Headline per task: AUROC of the out-of-fold score averaged over the 3 repeats, with a 95% patient-bootstrap CI
  (2,000 resamples, vectorised weighted-rank AUROC `wauc` from `analysis/tilebudget/code/aggregate_tb.py`). The three
  per-repeat AUROCs are also reported. Paired bootstrap deltas: ABMIL - mean-pool and ABMIL - shuffled.
- Aneuploidy: Spearman rho of the averaged OOF prediction vs the score, bootstrap CI.
- **Sensitivity analyses (reported regardless of outcome):** (a) within-cohort AUROC (STAD only, ESCA only) of the pooled
  OOF scores, because WGD prevalence differs by organ; (b) site-grouped CV: the same 5-fold x 3 repeats with folds split by
  tissue source site so no site appears in both train and test (Howard et al. 2021) for WGD and MSI.
- Cross-cohort: ABMIL trained on all WGD-labelled STAD (3 seeds, scores averaged), AUROC on ESCA-adeno with bootstrap CI.

## Pre-registered decision rules
- **Pipeline check (positive control):** MSI ABMIL AUROC >= 0.75 with CI lower bound > 0.5. If this fails the pipeline
  is called broken and the CIN results are reported as uninterpretable rather than as nulls.
- **Primary (WGD, pooled ABMIL):**
  - *Positive:* CI lower bound > 0.5 **and** the paired delta vs shuffled ABMIL has CI lower bound > 0. Called *robust*
    only if the site-grouped AUROC CI and the STAD-only AUROC CI also exclude 0.5; otherwise reported as "positive but
    possibly confounded by site/organ".
  - *Null:* CI includes 0.5, or the shuffled delta CI includes 0.
  - *Failed:* the positive control fails, or fewer than 90% of the cohort slides yield features.
- **Prediction (written down now, may well be wrong):** WGD AUROC ~0.65 (plausible range 0.58-0.75), MSI ~0.85,
  CIN-vs-rest ~0.70, aneuploidy rho ~0.25; ABMIL >= mean-pool by a small margin; shuffled ~0.50; site-only
  baseline non-trivial (>0.55) for at least one task; STAD->ESCA transfer lower than in-distribution.

## Figures
1. AUROC per task with 95% CIs: ABMIL vs mean-pool vs shuffled (and site-only reference), Spearman panel for aneuploidy.
2. WGD attention heatmaps (repeat 0 OOF model): 2 correct high-confidence cases (one WGD+, one WGD-) and 1 confident
   error, each with its top-8 attended tiles.
3. Cohort/label flow diagram with counts.

## Status log
(append-only; dated entries)
- 2026-10-19: DESIGN.md written; no features or model outputs exist yet.
- 2026-10-19 (cohort build, labels only, no images): 413 STAD + 65 ESCA-adeno patients (90 ESCA squamous dropped);
  WGD labelled 396 + 58 (162 + 36 positive); MSI (MANTIS >= 0.4) 77/410 STAD, 99.7% concordant with SUBTYPE = STAD_MSI;
  CIN subtype is 56/58 positive in ESCA-adeno, so the pooled CIN task is driven almost entirely by STAD.
- 2026-10-19 (smoke, 5 slides): two bugs caught and fixed before launch: (1) the DX1 slide of TCGA-BR-4362 is a
  1600x1200 Aperio snapshot without resolution metadata -> rule added: slides without `openslide.mpp-x` are excluded
  (0 tiles) and counted in the flow diagram; (2) the embed script failed on that 0-tile slide's empty metadata.
  Equivalence check on a native-20x slide (TCGA-HF-7134): 88 tiles within 16 px of an old `extract_one.py` tile,
  median cosine 0.943 (p5 0.911) — consistent with the small offset + bf16, i.e. weights/normalisation are right.
  Embedding speed ~8-11 s per 2,000-tile slide on an L40S.
- 2026-10-19: full prep campaign launched (478 slides, CPU); embed (60 GPU groups of 8 slides) follows.
- 2026-10-19 (prep): 40 of 478 DX1 slides (all TCGA-STAD, sites BR 23 + CG 17) have no resolution metadata
  (low-resolution snapshots, <=10,000 px) and were excluded under the rule above; feature coverage 438/478 = 91.6%
  (>= 90% floor). These slides had features in the old `features_uni_v2` set only because `extract_one.py` defaults to
  0.25 um/px when the tag is missing.
- 2026-10-19 (training): the GPU queue was saturated, so training units were also raced on epyc CPUs (identical code,
  `dev = cpu` when no GPU); ABMIL took ~67 s per fold on 8 cores. Embedding remained GPU-only.
- 2026-10-19 (results, see results/main.json): positive control passed; WGD call by the pre-registered rule = "positive
  (robust)". Predictions vs outcome: WGD 0.78 (predicted ~0.65, outside the written range 0.58-0.75 on the high side);
  MSI 0.88 (~0.85); CIN 0.86 (~0.70, higher); aneuploidy rho 0.53 (~0.25, higher); ABMIL >= mean-pool (yes, small);
  shuffled ~0.5 (yes); site-only > 0.55 for at least one task (yes, CIN 0.58); STAD->ESCA lower (yes, 0.57, CI spans 0.5).
  Not anticipated: within ESCA-adeno the pooled model has no WGD signal (0.46, n = 58).
- 2026-10-19 (POST HOC, not pre-registered; results/posthoc.json): to check that STAD WGD is not just subtype
  detection, WGD AUROC within STAD-CIN tumours (n = 190) and within MSS STAD (n = 295).
