"""Write docs/aceb_primary_killcoyne.md (one page) from the frozen package bundle and the development estimate (docs/paper_plan_killcoyne_cv.md item 4)."""
import json
FZ = json.load(open("models/killcoyne_frozen_pkg_v1/MANIFEST.json")); D = json.load(open("results/paper_final/killcoyne_cv.json"))["item4_development_estimate"]
txt = f"""# ACE-B primary analysis: fusion vs copy number under the Killcoyne protocol

Fixed before any ACE-B slide, CNV profile or label is processed. Written as item 4 of `docs/paper_plan_killcoyne_cv.md`. Any change is an amendment below the line, never an edit of this text. This analysis is the ACE-B primary; everything else, including `docs/aceb_analysis_plan.md` §4, is secondary (amendment 1 there).

**Hypothesis.** On NDBE samples, the per-sample AUROC of the frozen L-LATE (package features) is higher than that of the frozen L-CNV (package features).

**Models.** `models/killcoyne_frozen_pkg_v1/`, bundle SHA-256 `{FZ['bundle_sha256']}` (per-file hashes in its `MANIFEST.json`). L-CNV = glmnet on package CNV features; L-IMG = glmnet on the UNI2 slide-mean embedding; L-LATE = (L-CNV probability + L-IMG probability) / 2. Trained on SWG set C (676 samples, 80 patients). No ACE-B data is used for training, tuning, scaling, calibration or thresholds.

**Preprocessing.** Exactly as the bundle's `model_info.json`: CNV from QDNAseq 50 kb hg38 counts through the package segmentation with the hg38 adaptations of `scripts/paper_plan/kr_package.R`, tiles and arms scaled with the set-C constants in the bundle; image from UNI2 (tile 224, level 2, 256 tiles per slide) mean embeddings scaled with the bundle's set-C constants. Every sample the package segments is scored, whatever its QC flag; the QC-fail count is reported.

**Population and label.** ACE-B samples with both a CNV profile and a slide embedding whose sample histology is NDBE. Label per sample = the patient's progression status (1 if the patient progressed to HGD/IMC by the cohort owner's adjudication, else 0), shared by all of a patient's samples, as in the Killcoyne protocol and set C. Patients are not excluded for prevalent disease, since set C did not exclude them; this differs from `docs/aceb_analysis_plan.md` §3 and is stated here in advance.

**Statistic and test.** Δ = per-sample rank AUROC of L-LATE minus that of L-CNV, on the NDBE samples. 2,000 patient-bootstrap resamples (`numpy.random.RandomState(0)`; patients drawn with replacement with all their NDBE samples; resamples with one class skipped). One-sided, in favour of fusion: the 95% lower bound is the 5th percentile of the bootstrap Δ; one-sided p = (1 + #{{bootstrap Δ ≤ 0}}) / 2,001. The primary result is positive if the lower bound is above 0. Reported with n samples, n patients and n progressor patients.

**Secondary (all descriptive).** Patient-max AUROC; all samples; before the first HGD/IMC; L-EARLY, L-IMG alone; two-sided intervals; every analysis in `docs/aceb_analysis_plan.md`.

**Development estimate (context only, not a prediction).** Set C, stratified 10-fold × 10 CV, NDBE samples ({D['n_samples']} samples, {D['n_patients']} patients, {D['n_P_patients']} P): Δ {D['delta']:+.3f}, one-sided 95% lower bound {D['one_sided_95_lower']:+.3f}. ACE-B's size is not known at the time of writing, so its power is not stated.

---
Amendments: none.
"""
open("docs/aceb_primary_killcoyne.md", "w").write(txt); print("written", len(txt.split()), "words")
