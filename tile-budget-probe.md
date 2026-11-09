# How many tiles does a slide-level model need? A public-data probe

Rehan Zuberi, October 2026. Public data only (TCGA, CAMELYON16). Pre-registered design: [`analysis/outreach_tilebio/DESIGN.md`](analysis/outreach_tilebio/DESIGN.md); code in the same folder.

**Question.** Whole-slide images hold thousands of tiles each. If a model only needs 16–256 of them, embedding a slide costs 10–100x less. Does that hold, does it depend on the foundation model, and does it break when the lesion is small?

**Setup.** TCGA lung adenocarcinoma vs squamous (500 patients, patient-level 5-fold CV) and CAMELYON16 lymph-node metastasis (official split, 129 test slides). Every tissue tile (9.6 M in all) embedded with UNI2-h and Virchow2; gated-attention MIL vs a mean-pool baseline, 3 seeds; test budgets of 1–256 tiles chosen at random or by attention; 2,000-resample paired bootstrap.

**Results.**
- Diffuse task (TCGA): 256 random tiles match all tiles for both encoders (ΔAUROC +0.001 and −0.003), at 31x less GPU time per slide. Sixteen tiles cost a small but clear amount.
- Focal task (CAMELYON16): random sampling fails. 256 random tiles still lose 0.08 AUROC, and on micro-metastases no budget below all tiles matches using every tile. A single attention-selected tile matches all tiles, but attention needs every tile embedded first, so it shows where the signal is rather than saving compute.
- The two encoders agree within one budget step. Mean pooling matches attention on TCGA and fails on CAMELYON16 (0.59).

**What went wrong.** The first run gave chance-level results. A plain scikit-learn baseline on the same features located the fault: a file reader advancing by bytes as if they were fp16 elements. All 24 training units were rerun after the fix.
