#!/usr/bin/env python3
"""Item 21 scaffold (24 Sep 2026): ACE-B zero-shot validation of the frozen SWG models. Runs when three inputs exist:
  ACEB_SLIDES   dir of scanned H&E (ndpi/svs/tiff)                    -> UNI2 tiles at 0.5 um/px, 224 px (identical to SWG release)
  ACEB_CNV      table of arm-level copy number re-binned to the SWG 50 kb / 0.4x representation (same columns as
                feature_views/cnv/features_arms.csv), one row per sample_id
  ACEB_LABELS   table sample_id, patient_id, y_progressor (Leanne's adjudicated progression), seattle_grade
Thresholds are FIXED from SWG: results/numbers/swg_operating_points_patient.json (sens 0.95 / 1.0 rows) and the
other-fold thresholds in swg_robustness.json item14. No training, no tuning. Outputs AUROC/AUPRC with patient bootstrap,
specificity at the SWG-fixed thresholds, and calibration in the ACE-B prevalence.
STATUS: scaffold only; every step below raises NotImplementedError until the inputs arrive."""
import os, sys, json
REQUIRED = ["ACEB_SLIDES", "ACEB_CNV", "ACEB_LABELS"]
missing = [k for k in REQUIRED if not os.environ.get(k)]
if missing: sys.exit(f"ACE-B inputs not available yet: {missing}. See docstring.")
def featurise_slides(slide_dir): raise NotImplementedError("reuse campaigns/allslides/extract_worker.py logic with MAX_TILES=8000, 0.5 um/px")
def rebin_cnv(cnv_table): raise NotImplementedError("downsample 7x reads / re-bin to 50 kb, then arm-level features as feature_views/cnv/features_arms.csv")
def apply_frozen_models(): raise NotImplementedError("load training_final_nested_cv_v1/<family>/fold*/model.pt; average the 5 fold models' outputs per sample")
def score_at_fixed_thresholds(): raise NotImplementedError("thresholds from results/numbers/swg_operating_points_patient.json")
