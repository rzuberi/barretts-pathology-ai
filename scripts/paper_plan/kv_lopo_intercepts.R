# Killcoyne stratified CV, item 2 (docs/paper_plan_killcoyne_cv.md @ 76879b8): recover the intercept of every LOPO model by refitting the LOPO
# training set's glmnet path (deterministic) and reading the coefficient at the lambda already chosen (class_min); the recomputed prediction is
# compared with the stored one. ARM_ID indexes the LOPO arms below.
source("/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/scripts/paper_plan/kv_common.R")
arms <- data.frame(name = c("L-CNV their", "L-CNV pkg", "L-IMG fold-honest", "L-EARLY fold-honest their", "L-EARLY fold-honest pkg", "L-GRADE", "L-CNV+GRADE their", "L-CNV+GRADE pkg", "L-EARLY+GRADE their", "L-EARLY+GRADE pkg"),
                   dir = c("linear", "linear", rep("checks/linear", 8)), cfg = c(0, 4, 0, 1, 2, 3, 4, 5, 6, 7),
                   model = c("cnv", "cnv", "img", "early", "early", "grade", "cnvgrade", "cnvgrade", "earlygrade", "earlygrade"), cnvsrc = c("their", "pkg", "none", "their", "pkg", "none", "their", "pkg", "their", "pkg"),
                   img = c("fold", "fold", "fold", "fold", "fold", "fold", "fold", "fold", "whole", "whole"), stringsAsFactors = FALSE)   # checks grade arms used whole-set image scaling
a <- as.integer(Sys.getenv("ARM_ID", "0")); A <- arms[a + 1, ]; OD <- file.path(M, "cv", "lopo_intercepts"); dir.create(OD, recursive = TRUE, showWarnings = FALSE)
outf <- file.path(OD, sprintf("arm_%02d.csv", a)); if (file.exists(outf)) quit(save = "no")
P <- bind_rows(lapply(list.files(file.path(M, A$dir), sprintf("^cfg_%02d_chunk_.*csv$", A$cfg), full.names = TRUE), read_csv, col_types = cols(Sample = "c", Patient = "c"))) %>% filter(rule == "class_min")
feat <- make_features(A$model, A$cnvsrc, A$img); out <- list()
for (p in sort(unique(pat))) {
  te <- pat == p; tr <- which(!te); X <- feat(tr); lam <- unique(P$lambda[P$Patient == p]); stopifnot(length(lam) == 1)
  fit <- glmnet(X[tr, ], y[tr], family = "binomial", alpha = 0.9, standardize = FALSE); b0 <- as.numeric(coef(fit, s = lam)[1, 1])
  pr <- predict(fit, X[te, , drop = FALSE], s = lam, type = "response")[, 1]; st <- P$pred[match(ids[te], P$Sample)]
  out[[p]] <- data.frame(arm = A$name, Patient = p, label = max(y[te]), lambda = lam, intercept = b0, train_prev = mean(y[tr]), max_abs_pred_diff = max(abs(pr - st)))
}
write_csv(bind_rows(out), outf); message("KV LOPO INTERCEPTS DONE ", A$name)
