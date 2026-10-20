# Public tile-budget probe (outreach: TileBio) — design

Written 2026-10-19, before any tile was read or any model trained. Predictions and the primary endpoint below are fixed;
later changes go in the status log at the bottom with a reason and a timestamp, never by editing this section.

## Question
A private biopsy study (not reused here; only its headline is cited as motivation) found that 16 attention-selected tiles
nearly matched all tiles (AUROC 0.945 vs 0.952) and that 256 random tiles also nearly matched. Three questions on public slides:
1. Does it replicate?  2. Does it depend on the foundation model?  3. Does it break when the lesion is small?
Why it matters: if 16–256 tiles suffice, embedding cost per slide falls by 10–100x.

## Data (public only)
- **Diffuse task:** TCGA-LUAD vs TCGA-LUSC, GDC open access, one primary diagnostic slide (`-01Z-00-DX*`, lowest DX number) per patient;
  956 patients (478 / 478). Patient-level stratified 5-fold CV (seed 20261019); out-of-fold predictions pooled.
- **Focal task:** CAMELYON16 lymph-node metastasis, tumour vs normal, official split: train 270 (111 tumour / 159 normal),
  test 129 (49 tumour: 22 macro, 27 micro; 80 normal). Labels and macro/micro classes from the bucket's `evaluation/reference.csv`.
- No ERIN, OCCAMS, SWG, NHS or controlled-access data, labels or models. The optional TCGA-STAD/ESCA second diffuse task is dropped:
  `analysis/outreach_panakeia/` does not exist.

## Tiles and features
- Tissue tiles only, 224 x 224 px at 0.5 µm/px ("20x"): read at the nearest pyramid level with mpp ≤ 0.525 and resized to 224 when
  that level is finer (TCGA 40x scans: 448 px at level 0). Tissue = Otsu on the HSV saturation of a thumbnail (median-blurred,
  floor 15, V > 40 to drop black scanner background); a tile is kept if ≥ 25 % of its footprint is tissue. No per-slide tile cap.
- Phase 1 (CPU, one task per slide): download, tissue mask, read all tissue tiles into a uint8 array on disk.
- Phase 2 (GPU, one task per slide): embed every tile with UNI2-h (1,536-d CLS) and Virchow2 (2,560-d = CLS ⊕ mean of patch tokens),
  fp16 autocast. CONCH v1.5 only if time remains.

## Factors
- Foundation model: UNI2-h, Virchow2.
- Training bag: ALL (every tissue tile of the slide, full bag each step) vs R256 (a fixed random 256 tiles per slide).
  Reason for changing from the private study's "800 resampled per step": on CAMELYON16 a micro-metastasis can cover a handful of
  tiles, so an 800-tile subsample would often contain no tumour and the bag label would be wrong.
- Test budget m ∈ {1, 4, 16, 64, 256, all}; selection: random (3 independent draws, AUROC averaged over draws — each draw costs m tiles,
  so predictions are not averaged across draws), attention-top, attention-bottom. Attention = the trained model's own attention over all
  tiles of the test slide; attention-top therefore needs every tile embedded and is reported as an upper bound on informed selection,
  not as a saving.
- Aggregator: ABMIL (gated attention, the private study's architecture and hyperparameters: 25 epochs, Adam 1e-4, wd 1e-5, 16 bags per
  step by gradient accumulation, class-balanced BCE); baseline mean-pool + logistic regression on the same bags. 3 seeds; predictions
  averaged over seeds.
- Units: TCGA 2 models x 2 arms x 5 folds = 20 GPU units; CAMELYON16 2 x 2 x 1 = 4.

## Endpoints
- **Primary (per task x model, ABMIL trained on ALL):** the smallest random test budget m whose paired AUROC difference to m = all has a
  95 % bootstrap CI containing 0 (or lying above 0); the same for attention-top. If no m < all qualifies, the endpoint is "all".
- Bootstrap: 2,000 resamples of patients (TCGA; one slide per patient) or test slides (CAMELYON16), percentile CIs, paired deltas computed
  on the same resamples; vectorised weighted-rank AUROC (`wauc`, from the private study's aggregation code).
- **Secondary:** R256 vs ALL training at each test budget; mean-pool LR vs ABMIL; CAMELYON16 macro vs micro (AUROC of each lesion class
  vs all 80 normals) across budgets; fraction of attention-top-16 tiles inside the tumour annotation; embedding cost.
- **Cost:** GPU-seconds per slide (CUDA-synchronised forward time incl. host-to-device copy and normalisation, excluding disk reads and
  model loading) for all tiles vs 16 and 256 tiles, per model and per GPU type, measured on every slide; CPU tile-reading time logged.

## Predictions (pre-registered)
- P1 Diffuse (TCGA): random endpoint 16–64 tiles for both models; attention-top endpoint ≤ 16.
- P2 Focal (CAMELYON16): random endpoint much larger than on TCGA (≥ 256 or all); attention-top endpoint ≤ 16.
- P3 Small lesions: at m ≤ 64 random tiles the micro-metastasis AUROC falls further below its all-tile value than the macro AUROC does.
- P4 The ordering of budgets does not depend on the encoder (UNI2-h vs Virchow2 endpoints within one budget step of each other).
"Replicates" = P1 holds for both models. Anything else is reported as it comes out.

## Compute plan
Smoke first: 5 slides per dataset through phases 1 and 2, output counts verified, throughput measured. Budget ~1 GPU-day.
If measured throughput puts phase 2 above ~20 GPU-hours, TCGA is subsampled (stratified random, fixed seed) before any training; CAMELYON16
is never subsampled. Phase 3 packs features once per dataset x model (fp16, sequential reads); phase 4 GPU units hold the pack in host RAM.
All jobs through `scripts/cluster/campaign.sh` (h200 always with a cuda twin, `--exclude=clust1-h200-1`, virchow2 env for torch).

## Status log
- 2026-10-19: design written; manifests built (TCGA 956 patients, CAMELYON16 399 slides, 160 annotations); no tiles read yet.
- 2026-10-19 16:10: phase-1 smoke 10/10 slides tiled (TCGA 1,657-29,341 tiles, CAMELYON16 1,971-7,288). Compute fix only: tile reads moved
  from forked processes to threads (forked readers ran at ~4 tiles/s on some slides; threads 42 s for the same slide). Mask QC: annotated
  tumour area covered by kept tiles 0.997 / 0.938 / 1.000 on test_001 / test_004 / tumor_001 -> tissue rule kept unchanged.
- 2026-10-19 16:10, before any result: figure 3 slide rule fixed = the CAMELYON16 test tumour slide with the median tumour-tile fraction
  (lower median), UNI2-h, attention averaged over 3 seeds.
- 2026-10-19 16:25, before any training or result: embedding smoke 10/10 (UNI2-h + Virchow2, L40S). Measured ~3.4 ms per tile per model
  (L40S), so all 956 TCGA patients + CAMELYON16 projected ~25 L40S-hours > 20 -> pre-registered rule applied: TCGA subsampled to
  250 LUAD + 250 LUSC (stratified random, seed 20261019, original folds kept; code/subsample.py, column use in manifests/slides.csv).
  CAMELYON16 complete (399). Analysis set = 899 slides. GPU jobs run as pull workers (models loaded once, next slide prefetched) to remove
  ~40 s per-slide read/load overhead; per-slide outputs and locks unchanged. Virchow2 now loaded from explicit config + safetensors.
- 2026-10-20 12:05: BUG (mine), found before any interpretation: the first full run gave chance-level CAMELYON16 (0.52) and TCGA 0.71-0.76 for
  both ABMIL and mean-pool, while a plain sklearn mean-pool CV on the same features gave 0.957/0.960 (TCGA). Cause: the sequential pack reader
  in train_eval_pub.py advanced its offset by the BYTES returned by readinto() as if they were fp16 ELEMENTS, so beyond the first 512 MB rows
  were scrambled across slides. Fixed (byte view + assert on total bytes); features, tiles and the design are unchanged. All 24 training
  units rerun; the buggy predictions are kept in superseded_readinto_bug/ and not used. (Tile reader in embed_slide.py was unaffected: uint8.)
- 2026-10-20 12:05: three CAMELYON16 downloads (normal_089, test_042, test_126) stalled at ~0.25 MB/s and hit the 6 h worker walltime;
  resumed with HTTP Range, all 399 slides tiled and embedded. Cost tables restricted to the 899 analysis slides.
- 2026-10-20 12:24: rerun complete (24/24 units, 72 prediction files); aggregate in results/results.json. Outcome against the predictions:
  P1 (TCGA random endpoint 16-64 and attention-top <= 16, both encoders): NOT met. Random endpoint 64 (UNI2-h) and 256 (Virchow2);
  attention-top endpoint 64 and 256 (deltas at 16 tiles are -0.006 and -0.005 but the CIs exclude 0). By the definition above the
  private-study result does not replicate on TCGA, although random 256 matches all tiles for both encoders.
  P2 (CAMELYON16 random endpoint >= 256 or all; attention-top <= 16): met. Random endpoint = all for both; attention-top endpoint = 1.
  P3 (micro falls further below all tiles than macro at m <= 64 random): met. Random 64: macro -0.069 / -0.067, micro -0.220 / -0.240.
  P4 (encoder endpoints within one budget step): met (TCGA 64 vs 256; CAMELYON16 identical).
  Figure 3 slide by the fixed rule: test_079 (micro), 16/16 attention-top tiles inside the annotation.
