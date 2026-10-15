# Horizons (docs/paper_survival_horizons.md @ ee51db8): model fits under the stratified 10-fold x 10 CV of kv_cv.R.
# MODE=outer CFG=inter_their|inter_pkg|gradeagesex REP=1..10: new outer-CV arms (H1), output like kv_cv.R.
# MODE=inner CFG=cnv_their|cnv_pkg|img|early_their|early_pkg|inter_their|inter_pkg REP=1..10 FOLD=1..10: inner 5-fold OOF predictions for the
# outer training samples of (REP, FOLD) (H2 (a) thresholds, H3 stacks); inner folds stratified as the outer CV with set.seed(1000*REP+FOLD);
# lambda = class-error min of a 1 x 5 patient-grouped CV (set.seed(1)) on the inner training patients; scaling and PCA on the inner training rows.
source("/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/scripts/paper_plan/kv_common.R")
MODE <- Sys.getenv("MODE"); CFG <- Sys.getenv("CFG"); REP <- as.integer(Sys.getenv("REP", "1")); FOLD <- as.integer(Sys.getenv("FOLD", "0"))
HZ <- file.path(M, "horizons"); OD <- file.path(HZ, MODE); dir.create(OD, recursive = TRUE, showWarnings = FALSE)
outf <- if (MODE == "outer") file.path(OD, sprintf("%s_rep_%02d.csv", CFG, REP)) else file.path(OD, sprintf("%s_rep_%02d_fold_%02d.csv", CFG, REP, FOLD))
if (file.exists(outf)) quit(save = "no")
SM <- read_csv(file.path(HZ, "samples.csv"), col_types = cols(Sample = "c", .default = "c")); SM <- SM[match(ids, SM$Sample), ]; stopifnot(all(SM$Sample == ids))
AS <- cbind(age = as.numeric(SM$age), sex = as.numeric(SM$sex_M)); rownames(AS) <- ids; stopifnot(all(is.finite(AS)))
pca_block <- function(B, tr, k = 32) { pc <- prcomp(B[tr, ], center = TRUE, scale. = FALSE, rank. = k); Z <- predict(pc, B); zs(Z, tr) }
src <- sub("^[a-z]+_", "", CFG); mdl <- sub("_.*$", "", CFG)
CN <- if (src %in% c("their", "pkg")) readblock(sprintf("cnv_%s_C.csv", src)) else NULL
IMraw <- if (mdl %in% c("img", "early", "inter")) load_img() else NULL
feat <- function(tr) switch(mdl, cnv = CN, img = zs(IMraw, tr), early = cbind(CN, zs(IMraw, tr)), inter = cbind(pca_block(CN, tr), pca_block(zs(IMraw, tr), tr)),
                            gradeagesex = cbind(GR, zs(AS, tr)))
cvcurves1 <- function(X, yy, pp) {
  fit <- glmnet(X, yy, family = "binomial", alpha = 0.9, standardize = FALSE); lam <- fit$lambda
  fo <- shuffle_groups(pp, 1)[pp]; fe <- matrix(NA, 5, length(lam)); w <- numeric(5)
  for (k in 1:5) { te <- fo == k; f <- glmnet(X[!te, ], yy[!te], family = "binomial", alpha = 0.9, lambda = lam, standardize = FALSE); pr <- predict(f, X[te, , drop = FALSE], type = "response")
    if (ncol(pr) < length(lam)) pr <- cbind(pr, matrix(pr[, ncol(pr)], nrow(pr), length(lam) - ncol(pr)))
    fe[k, ] <- colMeans((pr > 0.5) != yy[te]); w[k] <- sum(te) }
  cm <- colSums(fe * w) / sum(w); list(fit = fit, class_min = max(lam[cm <= min(cm)])) }
strat_folds <- function(pp, yy, seed, nf) { pl <- tapply(yy, pp, max); Pp <- names(pl)[pl == 1]; Np <- names(pl)[pl == 0]; set.seed(seed); ord <- c(sample(Pp), sample(Np)); setNames(rep(1:nf, length.out = length(ord)), ord) }
fold_of <- strat_folds(pat, y, REP, 10)   # identical to kv_cv.R: set.seed(REP); P shuffled then NP; cyclic
out <- list()
if (MODE == "outer") {
  for (k in 1:10) {
    te <- fold_of[pat] == k; tr <- which(!te); X <- feat(tr); cv <- cvcurves(X[tr, ], y[tr], pat[tr])
    pr <- predict(cv$fit, X[te, , drop = FALSE], s = cv$class_min, type = "response")[, 1]
    out[[k]] <- data.frame(Sample = ids[te], Patient = pat[te], y = y[te], model = mdl, cnvsrc = src, rep = REP, fold = k, lambda = cv$class_min, pred = pr)
  }
} else {
  otr <- which(fold_of[pat] != FOLD); ifo <- strat_folds(pat[otr], y[otr], 1000 * REP + FOLD, 5)[pat[otr]]
  for (j in 1:5) {
    tri <- otr[ifo != j]; tei <- otr[ifo == j]; X <- feat(tri); cv <- cvcurves1(X[tri, ], y[tri], pat[tri])
    pr <- predict(cv$fit, X[tei, , drop = FALSE], s = cv$class_min, type = "response")[, 1]
    out[[j]] <- data.frame(Sample = ids[tei], Patient = pat[tei], y = y[tei], model = mdl, cnvsrc = src, rep = REP, fold = FOLD, inner_fold = j, lambda = cv$class_min, pred = pr)
  }
}
write_csv(bind_rows(out), outf); message("HZ FIT DONE ", MODE, " ", CFG, " rep ", REP, " fold ", FOLD)
