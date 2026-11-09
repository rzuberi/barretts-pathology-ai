# Two papers, agent CNC (docs/two_papers_CNC.md): the L-CNV recipe refitted on a new copy-number representation under the stored folds and lambda rules.
# A copy of tp_art_refit.R (= ex_e3_refit.R) with only the input / output directories changed and one switch added: kv_common.R sourced unchanged (set C,
# zs, cvcurves = the kv_cv.R outer lambda rule), the kv_cv.R outer loop and the hz_fit.R inner loop (cvcurves1, strat_folds copied verbatim).
# FEAT = their | pkg | pkgps | pkgpast (stored, already standardised blocks: used as is, SCALE=0, so 'their' must reproduce cv/preds/cfg_00 and
# horizons/inner/cnv_their_*) or killcoyne_mm/two_papers_CNC/feats/<FEAT>.csv.gz (z-scaled on training rows, SCALE=1; missing rows imputed with
# training-row means first). FEAT REP=1..10 -> killcoyne_mm/two_papers_CNC/refit/<FEAT>/{outer,inner}_rep_XX.csv (row-level, cluster only).
source("/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/scripts/paper_plan/kv_common.R")
FEAT <- Sys.getenv("FEAT"); ALT <- Sys.getenv("ALT", ""); REP <- as.integer(Sys.getenv("REP")); COEF <- Sys.getenv("COEF", "0") == "1"
E3 <- file.path(M, "two_papers_CNC"); OD <- file.path(E3, "refit", FEAT); dir.create(OD, recursive = TRUE, showWarnings = FALSE)
fo_ <- file.path(OD, sprintf("outer_rep_%02d.csv", REP)); fi_ <- file.path(OD, sprintf("inner_rep_%02d.csv", REP)); fc_ <- file.path(OD, sprintf("coef_rep_%02d.csv.gz", REP))
if (file.exists(fo_) && file.exists(fi_)) quit(save = "no")
rdblk <- function(f) { R <- read_csv(f, col_types = cols(Sample = "c", .default = "d"), progress = FALSE); X <- as.matrix(R[, -1]); rownames(X) <- R$Sample; stopifnot(all(ids %in% rownames(X))); X[ids, , drop = FALSE] }
STORED <- c(their = "cnv_their_C.csv", pkg = "cnv_pkg_C.csv", pkgps = "review_round4/r36_cnv_pkgps_C.csv", pkgpast = "review_round4/r36_cnv_pkgpast_C.csv")
RAW <- if (FEAT %in% names(STORED)) readblock(STORED[[FEAT]]) else rdblk(file.path(E3, "feats", paste0(FEAT, ".csv.gz")))
SCALE <- Sys.getenv("SCALE", if (FEAT %in% names(STORED)) "0" else "1") == "1"   # stored blocks are already standardised (kv_cv.R uses them as is)
XALT <- if (nzchar(ALT)) rdblk(file.path(E3, "feats", paste0(ALT, ".csv.gz"))) else NULL
if (!is.null(XALT)) stopifnot(ncol(XALT) == ncol(RAW))
# zs (kv_common.R) with its training-row centre and scale returned, after training-mean imputation of missing rows
scale_tr <- function(Mx, ref) {
  if (!SCALE) return(list(Z = Mx, mu = rep(0, ncol(Mx)), sdv = rep(1, ncol(Mx)), mi = rep(0, ncol(Mx))))
  mi <- colMeans(Mx[ref, , drop = FALSE], na.rm = TRUE); if (anyNA(Mx)) { w <- which(is.na(Mx), arr.ind = TRUE); Mx[w] <- mi[w[, 2]] }
  mu <- colMeans(Mx[ref, , drop = FALSE]); sdv <- apply(Mx[ref, , drop = FALSE], 2, sd); Z <- sweep(sweep(Mx, 2, mu), 2, ifelse(sdv > 0, sdv, 1), "/"); Z[, sdv == 0] <- 0
  list(Z = Z, mu = mu, sdv = sdv, mi = mi) }
apply_sc <- function(Mx, sc) { if (anyNA(Mx)) { w <- which(is.na(Mx), arr.ind = TRUE); Mx[w] <- sc$mi[w[, 2]] }
  Z <- sweep(sweep(Mx, 2, sc$mu), 2, ifelse(sc$sdv > 0, sc$sdv, 1), "/"); Z[, sc$sdv == 0] <- 0; Z }
rawcoef <- function(fit, s, sc) { b <- as.numeric(as.matrix(coef(fit, s = s))); w <- ifelse(sc$sdv > 0, b[-1] / ifelse(sc$sdv > 0, sc$sdv, 1), 0); c(b[1] - sum(w * sc$mu), w) }
# --- copied verbatim from hz_fit.R (inner lambda rule and stratified fold assignment)
cvcurves1 <- function(X, yy, pp) {
  fit <- glmnet(X, yy, family = "binomial", alpha = 0.9, standardize = FALSE); lam <- fit$lambda
  fo <- shuffle_groups(pp, 1)[pp]; fe <- matrix(NA, 5, length(lam)); w <- numeric(5)
  for (k in 1:5) { te <- fo == k; f <- glmnet(X[!te, ], yy[!te], family = "binomial", alpha = 0.9, lambda = lam, standardize = FALSE); pr <- predict(f, X[te, , drop = FALSE], type = "response")
    if (ncol(pr) < length(lam)) pr <- cbind(pr, matrix(pr[, ncol(pr)], nrow(pr), length(lam) - ncol(pr)))
    fe[k, ] <- colMeans((pr > 0.5) != yy[te]); w[k] <- sum(te) }
  cm <- colSums(fe * w) / sum(w); list(fit = fit, class_min = max(lam[cm <= min(cm)])) }
strat_folds <- function(pp, yy, seed, nf) { pl <- tapply(yy, pp, max); Pp <- names(pl)[pl == 1]; Np <- names(pl)[pl == 0]; set.seed(seed); ord <- c(sample(Pp), sample(Np)); setNames(rep(1:nf, length.out = length(ord)), ord) }
fold_of <- strat_folds(pat, y, REP, 10)   # identical to kv_cv.R: set.seed(REP); P shuffled then NP; cyclic
pa <- function(fit, s, sc, rows) if (is.null(XALT)) NA_real_ else predict(fit, apply_sc(XALT[rows, , drop = FALSE], sc), s = s, type = "response")[, 1]
outO <- list(); outI <- list(); CO <- list()
for (k in 1:10) {   # outer (kv_cv.R)
  te <- fold_of[pat] == k; tr <- which(!te); sc <- scale_tr(RAW, tr); X <- sc$Z; cv <- cvcurves(X[tr, ], y[tr], pat[tr])
  pr <- predict(cv$fit, X[te, , drop = FALSE], s = cv$class_min, type = "response")[, 1]
  outO[[k]] <- data.frame(Sample = ids[te], Patient = pat[te], y = y[te], feat = FEAT, rep = REP, fold = k, lambda = cv$class_min, pred = pr, pred_alt = pa(cv$fit, cv$class_min, sc, which(te)))
  if (COEF) CO[[length(CO) + 1]] <- c(REP, k, 0, rawcoef(cv$fit, cv$class_min, sc))
  otr <- which(fold_of[pat] != k); ifo <- strat_folds(pat[otr], y[otr], 1000 * REP + k, 5)[pat[otr]]   # inner (hz_fit.R)
  for (j in 1:5) {
    tri <- otr[ifo != j]; tei <- otr[ifo == j]; sc <- scale_tr(RAW, tri); X <- sc$Z; cv <- cvcurves1(X[tri, ], y[tri], pat[tri])
    pr <- predict(cv$fit, X[tei, , drop = FALSE], s = cv$class_min, type = "response")[, 1]
    outI[[length(outI) + 1]] <- data.frame(Sample = ids[tei], Patient = pat[tei], y = y[tei], feat = FEAT, rep = REP, fold = k, inner_fold = j, lambda = cv$class_min, pred = pr, pred_alt = pa(cv$fit, cv$class_min, sc, tei))
    if (COEF) CO[[length(CO) + 1]] <- c(REP, k, j, rawcoef(cv$fit, cv$class_min, sc))
  }
}
if (COEF) { CM <- do.call(rbind, CO); colnames(CM) <- c("rep", "fold", "inner_fold", "c0", paste0("w", seq_len(ncol(RAW)))); write_csv(as.data.frame(CM), fc_) }
write_csv(bind_rows(outI), fi_); write_csv(bind_rows(outO), fo_); message("TP CNC REFIT DONE ", FEAT, " rep ", REP)
