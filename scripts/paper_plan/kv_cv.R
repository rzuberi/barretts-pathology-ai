# Killcoyne stratified CV, items 1 and 3 (docs/paper_plan_killcoyne_cv.md @ 76879b8). CFG_ID = model x CNV source, REP = repeat seed 1..10.
# Outer: 10 patient-grouped folds stratified by P status (set.seed(REP); P shuffled then NP shuffled, concatenated, folds assigned cyclically).
source("/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/scripts/paper_plan/kv_common.R")
grid <- data.frame(model = c("cnv", "cnv", "img", "early", "early", "grade", "cnvgrade", "cnvgrade", "earlygrade", "earlygrade"), cnvsrc = c("their", "pkg", "none", "their", "pkg", "none", "their", "pkg", "their", "pkg"), stringsAsFactors = FALSE)
cfg <- as.integer(Sys.getenv("CFG_ID", "0")); rep_ <- as.integer(Sys.getenv("REP", "1")); g <- grid[cfg + 1, ]
OD <- file.path(M, "cv", "preds"); dir.create(OD, recursive = TRUE, showWarnings = FALSE); outf <- file.path(OD, sprintf("cfg_%02d_rep_%02d.csv", cfg, rep_)); if (file.exists(outf)) quit(save = "no")
pl <- tapply(y, pat, max); Pp <- names(pl)[pl == 1]; Np <- names(pl)[pl == 0]
set.seed(rep_); ord <- c(sample(Pp), sample(Np)); fold_of <- setNames(rep(1:10, length.out = length(ord)), ord)
feat <- make_features(g$model, g$cnvsrc, "fold"); out <- list()
for (k in 1:10) {
  te <- fold_of[pat] == k; tr <- which(!te); X <- feat(tr); cv <- cvcurves(X[tr, ], y[tr], pat[tr])
  pr <- predict(cv$fit, X[te, , drop = FALSE], s = cv$class_min, type = "response")[, 1]; b0 <- as.numeric(coef(cv$fit, s = cv$class_min)[1, 1])
  out[[k]] <- data.frame(Sample = ids[te], Patient = pat[te], y = y[te], cfg = cfg, model = g$model, cnvsrc = g$cnvsrc, rep = rep_, fold = k, lambda = cv$class_min, intercept = b0, train_prev = mean(y[tr]), pred = pr)
}
write_csv(bind_rows(out), outf); message("KV CV DONE cfg ", cfg, " rep ", rep_)
