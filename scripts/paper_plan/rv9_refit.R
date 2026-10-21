# Review round 9, B4 refits of the H&E model (docs/paper_review_round9_false_flags.md @ 7db8e14). Copies rv7_refit.R / rv4_refit.R for CFG=img on set C:
# kv_common.R is sourced unchanged; the kv_cv.R outer loop and the hz_fit.R inner loop (cvcurves1, strat_folds, same folds and lambda rule) are reproduced, with only the
# training rows or the observation weights changed per VAR. Row-level outputs -> killcoyne_mm/review_round9/refit/<VAR>/; stored predictions are never written.
# VAR=b4a  non-progressor sample weight 2 (glmnet weights, in the fit and in the inner lambda cross-validation)
# VAR=b4b  training on NDBE samples only (rows with Pathology NDBE; image block scaled on those rows); predictions for every row of the held-out patients
# VAR=b4w1 weight 1, all rows: the reproduction check against the stored predictions (cv/preds/cfg_02, horizons/inner/img_*)
# MODE=outer|inner REP=1..10 [FOLD=1..10; inner only, 0 = all ten]
source("/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/scripts/paper_plan/kv_common.R")
VAR <- Sys.getenv("VAR"); MODE <- Sys.getenv("MODE"); REP <- as.integer(Sys.getenv("REP", "1")); FOLDS <- as.integer(Sys.getenv("FOLD", "0"))
stopifnot(VAR %in% c("b4a", "b4b", "b4w1"), MODE %in% c("outer", "inner"))
OD <- file.path(M, "review_round9", "refit", VAR); dir.create(OD, recursive = TRUE, showWarnings = FALSE)
if (FOLDS == 0) FOLDS <- 1:10
outf <- function(fold) if (MODE == "outer") file.path(OD, sprintf("outer_img_rep_%02d.csv", REP)) else file.path(OD, sprintf("inner_img_rep_%02d_fold_%02d.csv", REP, fold))
todo <- if (MODE == "outer") !file.exists(outf(0)) else !file.exists(sapply(FOLDS, outf))
if (!any(todo)) quit(save = "no")
strat_folds <- function(pp, yy, seed, nf) { pl <- tapply(yy, pp, max); Pp <- names(pl)[pl == 1]; Np <- names(pl)[pl == 0]; set.seed(seed); ord <- c(sample(Pp), sample(Np)); setNames(rep(1:nf, length.out = length(ord)), ord) }
fold_of <- strat_folds(pat, y, REP, 10)   # identical to kv_cv.R
IMraw <- load_img(); feat <- function(tr) zs(IMraw, tr); ndbe <- C$Pathology == "NDBE"
wobs <- if (VAR == "b4a") ifelse(y == 0, 2, 1) else rep(1, n)
fitrows <- function(rows) if (VAR == "b4b") rows[ndbe[rows]] else rows
cvcurves_w <- function(X, yy, pp, w) {   # cvcurves (kv_common.R) with observation weights
  fit <- glmnet(X, yy, family = "binomial", alpha = 0.9, standardize = FALSE, weights = w); lam <- fit$lambda; cls <- matrix(NA, 10, length(lam))
  for (r in 1:10) {
    fo <- shuffle_groups(pp, r)[pp]; fe <- matrix(NA, 5, length(lam)); wk <- numeric(5)
    for (k in 1:5) { te <- fo == k; f <- glmnet(X[!te, ], yy[!te], family = "binomial", alpha = 0.9, lambda = lam, standardize = FALSE, weights = w[!te]); pr <- predict(f, X[te, , drop = FALSE], type = "response")
      if (ncol(pr) < length(lam)) pr <- cbind(pr, matrix(pr[, ncol(pr)], nrow(pr), length(lam) - ncol(pr)))
      fe[k, ] <- colMeans((pr > 0.5) != yy[te]); wk[k] <- sum(te) }
    cls[r, ] <- colSums(fe * wk) / sum(wk) }
  cm <- colMeans(cls); list(fit = fit, class_min = max(lam[cm <= min(cm)])) }
cvcurves1_w <- function(X, yy, pp, w) {   # cvcurves1 (hz_fit.R) with observation weights
  fit <- glmnet(X, yy, family = "binomial", alpha = 0.9, standardize = FALSE, weights = w); lam <- fit$lambda
  fo <- shuffle_groups(pp, 1)[pp]; fe <- matrix(NA, 5, length(lam)); wk <- numeric(5)
  for (k in 1:5) { te <- fo == k; f <- glmnet(X[!te, ], yy[!te], family = "binomial", alpha = 0.9, lambda = lam, standardize = FALSE, weights = w[!te]); pr <- predict(f, X[te, , drop = FALSE], type = "response")
    if (ncol(pr) < length(lam)) pr <- cbind(pr, matrix(pr[, ncol(pr)], nrow(pr), length(lam) - ncol(pr)))
    fe[k, ] <- colMeans((pr > 0.5) != yy[te]); wk[k] <- sum(te) }
  cm <- colSums(fe * wk) / sum(wk); list(fit = fit, class_min = max(lam[cm <= min(cm)])) }
if (MODE == "outer") {
  out <- list()
  for (k in 1:10) {
    te <- fold_of[pat] == k; tr <- which(!te); trf <- fitrows(tr); X <- feat(trf); cv <- cvcurves_w(X[trf, ], y[trf], pat[trf], wobs[trf])
    pr <- predict(cv$fit, X[te, , drop = FALSE], s = cv$class_min, type = "response")[, 1]; b0 <- as.numeric(coef(cv$fit, s = cv$class_min)[1, 1])
    out[[k]] <- data.frame(Sample = ids[te], Patient = pat[te], y = y[te], var = VAR, model = "img", cnvsrc = "none", rep = REP, fold = k, lambda = cv$class_min, intercept = b0, train_prev = mean(y[trf]), pred = pr)
  }
  write_csv(bind_rows(out), outf(0))
} else {
  for (FOLD in FOLDS[todo]) {
    otr <- which(fold_of[pat] != FOLD); ifo <- strat_folds(pat[otr], y[otr], 1000 * REP + FOLD, 5)[pat[otr]]; out <- list()
    for (j in 1:5) {
      tri <- otr[ifo != j]; tei <- otr[ifo == j]; trf <- fitrows(tri); X <- feat(trf); cv <- cvcurves1_w(X[trf, ], y[trf], pat[trf], wobs[trf])
      pr <- predict(cv$fit, X[tei, , drop = FALSE], s = cv$class_min, type = "response")[, 1]
      out[[j]] <- data.frame(Sample = ids[tei], Patient = pat[tei], y = y[tei], var = VAR, model = "img", cnvsrc = "none", rep = REP, fold = FOLD, inner_fold = j, lambda = cv$class_min, pred = pr)
    }
    write_csv(bind_rows(out), outf(FOLD))
  }
}
message("RV9 REFIT DONE ", VAR, " ", MODE, " rep ", REP)
