# Review round 4, Part B (docs/paper_review_round4_analyses.md @ 45b77bf): refits of the existing linear models under the existing folds. kv_common.R
# is sourced unchanged; the kv_cv.R outer loop and the hz_fit.R inner loop are reproduced verbatim (cvcurves1 and strat_folds copied from hz_fit.R), with
# only the copy-number block or the lambda rule changed as stated per MODE. Row-level outputs -> killcoyne_mm/review_round4/<dir>/; stored predictions
# are never written.
# MODE=outer CFG=cnv_pkg|cnv_pkgps|cnv_pkgpast REP=1..10: kv_cv.R outer for L-CNV on the package block named by CFG (cnv_pkg = the stored joint block,
#   a reproduction of cv/preds/cfg_01; cnv_pkgps = per-sample segmentation, cnv_pkgpast = past-only segmentation; rv4_reseg.R) -> r36_outer/.
# MODE=inner CFG REP FOLD: hz_fit.R inner (5-fold OOF predictions for the outer training rows of (REP, FOLD)) on that block -> r36_inner/.
# MODE=b5_inner CFG=cnv_their|img REP FOLD: hz_fit.R inner with, in addition, the lambda at the minimum binomial deviance of the same 1 x 5 inner curve
#   read off the same folds and fits (as F4, rv2_refit.R cvcurves_dev); columns lambda / pred are the class-error rule (reproduction of the stored inner
#   file), lambda_dev / pred_dev the deviance rule -> b5_inner/.
source("/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/scripts/paper_plan/kv_common.R")
MODE <- Sys.getenv("MODE"); CFG <- Sys.getenv("CFG"); REP <- as.integer(Sys.getenv("REP", "1")); FOLD <- as.integer(Sys.getenv("FOLD", "0"))
R4 <- file.path(M, "review_round4"); OD <- file.path(R4, c(outer = "r36_outer", inner = "r36_inner", b5_inner = "b5_inner")[[MODE]]); dir.create(OD, recursive = TRUE, showWarnings = FALSE)
outf <- if (MODE == "outer") file.path(OD, sprintf("%s_rep_%02d.csv", CFG, REP)) else file.path(OD, sprintf("%s_rep_%02d_fold_%02d.csv", CFG, REP, FOLD))
if (file.exists(outf)) quit(save = "no")
BLK <- c(cnv_their = "cnv_their_C.csv", cnv_pkg = "cnv_pkg_C.csv", cnv_pkgps = "review_round4/r36_cnv_pkgps_C.csv", cnv_pkgpast = "review_round4/r36_cnv_pkgpast_C.csv")
mdl <- sub("_.*$", "", CFG); src <- if (CFG == "img") "none" else sub("^cnv_", "", CFG)
# feature builder as kv_common.R make_features / hz_fit.R feat for model cnv (block as stored) and img (slide-mean embedding scaled on the training rows)
if (mdl == "img") { IMraw <- load_img(); feat <- function(tr) zs(IMraw, tr) } else { CN <- readblock(BLK[[CFG]]); feat <- function(tr) CN }
# --- copied verbatim from hz_fit.R (inner lambda rule and stratified fold assignment)
cvcurves1 <- function(X, yy, pp) {
  fit <- glmnet(X, yy, family = "binomial", alpha = 0.9, standardize = FALSE); lam <- fit$lambda
  fo <- shuffle_groups(pp, 1)[pp]; fe <- matrix(NA, 5, length(lam)); w <- numeric(5)
  for (k in 1:5) { te <- fo == k; f <- glmnet(X[!te, ], yy[!te], family = "binomial", alpha = 0.9, lambda = lam, standardize = FALSE); pr <- predict(f, X[te, , drop = FALSE], type = "response")
    if (ncol(pr) < length(lam)) pr <- cbind(pr, matrix(pr[, ncol(pr)], nrow(pr), length(lam) - ncol(pr)))
    fe[k, ] <- colMeans((pr > 0.5) != yy[te]); w[k] <- sum(te) }
  cm <- colSums(fe * w) / sum(w); list(fit = fit, class_min = max(lam[cm <= min(cm)])) }
strat_folds <- function(pp, yy, seed, nf) { pl <- tapply(yy, pp, max); Pp <- names(pl)[pl == 1]; Np <- names(pl)[pl == 0]; set.seed(seed); ord <- c(sample(Pp), sample(Np)); setNames(rep(1:nf, length.out = length(ord)), ord) }
# --- cvcurves1 with the binomial deviance read off the same folds and fits (B5; clamping as rv2_refit.R cvcurves_dev / km_linear.R)
cvcurves1_dev <- function(X, yy, pp) {
  fit <- glmnet(X, yy, family = "binomial", alpha = 0.9, standardize = FALSE); lam <- fit$lambda
  fo <- shuffle_groups(pp, 1)[pp]; fe <- fd <- matrix(NA, 5, length(lam)); w <- numeric(5)
  for (k in 1:5) { te <- fo == k; f <- glmnet(X[!te, ], yy[!te], family = "binomial", alpha = 0.9, lambda = lam, standardize = FALSE); pr <- predict(f, X[te, , drop = FALSE], type = "response")
    if (ncol(pr) < length(lam)) pr <- cbind(pr, matrix(pr[, ncol(pr)], nrow(pr), length(lam) - ncol(pr)))
    pc <- pmin(pmax(pr, 1e-5), 1 - 1e-5)
    fe[k, ] <- colMeans((pr > 0.5) != yy[te]); fd[k, ] <- colMeans(-2 * (yy[te] * log(pc) + (1 - yy[te]) * log(1 - pc))); w[k] <- sum(te) }
  cm <- colSums(fe * w) / sum(w); dm <- colSums(fd * w) / sum(w); list(fit = fit, class_min = max(lam[cm <= min(cm)]), dev_min = lam[which.min(dm)]) }
nnz <- function(fit, s) sum(as.numeric(as.matrix(coef(fit, s = s)))[-1] != 0)
fold_of <- strat_folds(pat, y, REP, 10)   # identical to kv_cv.R: set.seed(REP); P shuffled then NP; cyclic
out <- list()
if (MODE == "outer") {
  for (k in 1:10) {
    te <- fold_of[pat] == k; tr <- which(!te); X <- feat(tr); cv <- cvcurves(X[tr, ], y[tr], pat[tr])
    pr <- predict(cv$fit, X[te, , drop = FALSE], s = cv$class_min, type = "response")[, 1]; b0 <- as.numeric(coef(cv$fit, s = cv$class_min)[1, 1])
    out[[k]] <- data.frame(Sample = ids[te], Patient = pat[te], y = y[te], cfg = CFG, model = mdl, cnvsrc = src, rep = REP, fold = k, lambda = cv$class_min, intercept = b0, train_prev = mean(y[tr]), pred = pr)
  }
} else if (MODE == "inner") {
  otr <- which(fold_of[pat] != FOLD); ifo <- strat_folds(pat[otr], y[otr], 1000 * REP + FOLD, 5)[pat[otr]]
  for (j in 1:5) {
    tri <- otr[ifo != j]; tei <- otr[ifo == j]; X <- feat(tri); cv <- cvcurves1(X[tri, ], y[tri], pat[tri])
    pr <- predict(cv$fit, X[tei, , drop = FALSE], s = cv$class_min, type = "response")[, 1]
    out[[j]] <- data.frame(Sample = ids[tei], Patient = pat[tei], y = y[tei], model = mdl, cnvsrc = src, rep = REP, fold = FOLD, inner_fold = j, lambda = cv$class_min, pred = pr)
  }
} else if (MODE == "b5_inner") {
  stopifnot(CFG %in% c("cnv_their", "img"))
  otr <- which(fold_of[pat] != FOLD); ifo <- strat_folds(pat[otr], y[otr], 1000 * REP + FOLD, 5)[pat[otr]]
  for (j in 1:5) {
    tri <- otr[ifo != j]; tei <- otr[ifo == j]; X <- feat(tri); cv <- cvcurves1_dev(X[tri, ], y[tri], pat[tri])
    pc <- predict(cv$fit, X[tei, , drop = FALSE], s = cv$class_min, type = "response")[, 1]; pd <- predict(cv$fit, X[tei, , drop = FALSE], s = cv$dev_min, type = "response")[, 1]
    out[[j]] <- data.frame(Sample = ids[tei], Patient = pat[tei], y = y[tei], model = mdl, cnvsrc = src, rep = REP, fold = FOLD, inner_fold = j, lambda = cv$class_min, lambda_dev = cv$dev_min,
                           nnz_class = nnz(cv$fit, cv$class_min), nnz_dev = nnz(cv$fit, cv$dev_min), pred = pc, pred_dev = pd)
  }
} else stop("unknown MODE ", MODE)
write_csv(bind_rows(out), outf); message("RV4 REFIT DONE ", MODE, " ", CFG, " rep ", REP, " fold ", FOLD)
