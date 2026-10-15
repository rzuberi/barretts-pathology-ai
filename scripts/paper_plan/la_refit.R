# Latent space (docs/paper_latent_attention.md @ dfac9ad), Step 0 refit check. CFG = img | early_their | early_pkg | inter_their | inter_pkg, REP = 1..10.
# Refits every outer fold exactly as kv_cv.R (img, early) / hz_fit.R MODE=outer (inter) with the unchanged kv_common.R; compares with the stored out-of-fold
# predictions; writes per-row sums of the representation (img, early) or the 64-d representation itself (inter). Row-level output stays on the cluster.
source("/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/scripts/paper_plan/kv_common.R")
CFG <- Sys.getenv("CFG"); REP <- as.integer(Sys.getenv("REP")); OD <- file.path(M, "latent_attention", "refit"); dir.create(OD, recursive = TRUE, showWarnings = FALSE)
outc <- file.path(OD, sprintf("check_%s_rep_%02d.csv", CFG, REP)); if (file.exists(outc)) quit(save = "no")
mdl <- sub("_.*$", "", CFG); src <- if (mdl == "img") "none" else sub("^[a-z]+_", "", CFG)
if (mdl == "inter") {   # hz_fit.R:13,15-17 verbatim
  pca_block <- function(B, tr, k = 32) { pc <- prcomp(B[tr, ], center = TRUE, scale. = FALSE, rank. = k); Z <- predict(pc, B); zs(Z, tr) }
  CN <- readblock(sprintf("cnv_%s_C.csv", src)); IMraw <- load_img(); feat <- function(tr) cbind(pca_block(CN, tr), pca_block(zs(IMraw, tr), tr))
  stored <- read_csv(file.path(M, "horizons", "outer", sprintf("inter_%s_rep_%02d.csv", src, REP)), col_types = cols(Sample = "c"))
} else {
  cfg <- c(img = 2, early_their = 3, early_pkg = 4)[[CFG]]; feat <- make_features(mdl, src, "fold")
  stored <- read_csv(file.path(M, "cv", "preds", sprintf("cfg_%02d_rep_%02d.csv", cfg, REP)), col_types = cols(Sample = "c"))
}
pl <- tapply(y, pat, max); Pp <- names(pl)[pl == 1]; Np <- names(pl)[pl == 0]
set.seed(REP); ord <- c(sample(Pp), sample(Np)); fold_of <- setNames(rep(1:10, length.out = length(ord)), ord)   # kv_cv.R:7-8
pr_all <- setNames(rep(NA_real_, n), ids); fp <- list()
for (k in 1:10) {
  te <- fold_of[pat] == k; tr <- which(!te); X <- feat(tr); cv <- cvcurves(X[tr, ], y[tr], pat[tr])                  # kv_cv.R:11-12
  pr_all[te] <- predict(cv$fit, X[te, , drop = FALSE], s = cv$class_min, type = "response")[, 1]
  if (mdl == "inter") { D <- data.frame(Sample = ids, X, check.names = FALSE); colnames(D)[-1] <- sprintf("z%02d", 1:ncol(X)); write_csv(D, gzfile(file.path(OD, sprintf("repr_%s_rep_%02d_fold_%02d.csv.gz", CFG, REP, k)))) }
  else fp[[k]] <- data.frame(Sample = ids, fold = k, rowsum = rowSums(X), ncol = ncol(X))
}
st <- setNames(stored$pred, stored$Sample)[ids]; sf <- setNames(stored$fold, stored$Sample)[ids]
if (length(fp)) write_csv(bind_rows(fp), file.path(OD, sprintf("rowsum_%s_rep_%02d.csv", CFG, REP)))
write_csv(data.frame(cfg = CFG, rep = REP, n = sum(!is.na(st)), fold_match = all(sf == fold_of[pat]), max_abs_diff = max(abs(pr_all - st))), outc)
message("LA REFIT DONE ", CFG, " rep ", REP, " maxdiff ", signif(max(abs(pr_all - st)), 3))
