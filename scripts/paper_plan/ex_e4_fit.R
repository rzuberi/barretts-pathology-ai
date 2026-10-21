# Exploration E4 (docs/paper_explore_E4.md): time-aware refits under the stored patient-grouped folds. kv_common.R is sourced unchanged (set C, blocks,
# zs, shuffle_groups, cvcurves); folds as kv_cv.R / hz_fit.R (strat_folds(set C, REP, 10) outer; strat_folds(outer training, 1000*REP+FOLD, 5) inner).
# Survival times from ex_e4_prep.py (killcoyne_mm/explore_E4/surv.csv). Row-level outputs -> killcoyne_mm/explore_E4/fits/<CFG>/; stored predictions untouched.
# CFG = <kind>_<block>, block img|cnv; kind:
#   coxx  Cox elastic net, non-progressors censored at extended (DB) follow-up;  coxs  same, study follow-up (last sequenced endoscopy)
#   h3/h5 binary: progressor samples with event within H years vs non-progressor samples with >= H extended follow-up (others left out of training)
#   wt    binary, all samples, certainty weights (P: 1 if <= 8 y before event else 0.5; NP: clip(ext follow-up / 5, 0.5, 1.5))
#   coxp  (added after the first results) Cox, extended follow-up, trained on pre-event samples only (no at/after-event progressor samples)
#   wt2   (added after the first results) stronger weights: P 1 if <= 5 y, 0.5 if 5-8 y, 0.25 if > 8 y; NP clip(ext follow-up / 5, 0.25, 2)
# MODE=outer|inner REP=1..10 [FOLD=1..10 for inner; 0 = all]
source("/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/scripts/paper_plan/kv_common.R")
suppressMessages(library(survival))
CFG <- Sys.getenv("CFG"); MODE <- Sys.getenv("MODE"); REP <- as.integer(Sys.getenv("REP", "1")); FOLDS <- as.integer(Sys.getenv("FOLD", "0"))
kind <- sub("_.*$", "", CFG); blk <- sub("^[a-z0-9]+_", "", CFG); stopifnot(kind %in% c("coxx", "coxs", "coxp", "h3", "h5", "wt", "wt2"), blk %in% c("img", "cnv"), MODE %in% c("outer", "inner"))
OD <- file.path(M, "explore_E4", "fits", CFG); dir.create(OD, recursive = TRUE, showWarnings = FALSE)
if (FOLDS == 0) FOLDS <- 1:10
outf <- function(fold) if (MODE == "outer") file.path(OD, sprintf("outer_rep_%02d.csv", REP)) else file.path(OD, sprintf("inner_rep_%02d_fold_%02d.csv", REP, fold))
todo <- if (MODE == "outer") !file.exists(outf(0)) else !file.exists(sapply(FOLDS, outf))
if (!any(todo)) quit(save = "no")
SV <- read_csv(file.path(M, "explore_E4", "surv.csv"), col_types = cols(Sample = "c", Patient = "c", pre = "c", .default = "d")); SV <- SV[match(ids, SV$Sample), ]
stopifnot(all(SV$Sample == ids), all(SV$y == y))
tt <- if (kind == "coxs") SV$t_study else SV$t_ext; tp <- SV$t_study   # progressor event times are the same in both
if (blk == "img") { IMraw <- load_img(); feat <- function(tr) zs(IMraw, tr) } else { CN <- readblock("cnv_their_C.csv"); feat <- function(tr) CN }
iscox <- kind %in% c("coxx", "coxs", "coxp"); PREv <- SV$pre == "True"
# training rows used and weights, per kind
H <- c(h3 = 3, h5 = 5)[kind]
use <- function(rows) if (kind %in% c("h3", "h5")) rows[(y[rows] == 1 & tp[rows] <= H) | (y[rows] == 0 & SV$t_ext[rows] >= H)] else if (kind == "coxp") rows[PREv[rows]] else rows
wts <- function(rows) if (kind == "wt") ifelse(y[rows] == 1, ifelse(tp[rows] <= 8, 1, 0.5), pmin(1.5, pmax(0.5, SV$t_ext[rows] / 5))) else if (kind == "wt2") ifelse(y[rows] == 1, ifelse(tp[rows] <= 5, 1, ifelse(tp[rows] <= 8, 0.5, 0.25)), pmin(2, pmax(0.25, SV$t_ext[rows] / 5))) else rep(1, length(rows))
# ---- lambda selection: binomial = class-error rule of kv_common.R cvcurves / hz_fit.R cvcurves1 (weighted when kind == wt); cox = grouped partial-likelihood deviance
cv_bin <- function(X, yy, pp, ww, nrep) {
  fit <- glmnet(X, yy, family = "binomial", alpha = 0.9, standardize = FALSE, weights = ww); lam <- fit$lambda; cls <- matrix(NA, nrep, length(lam))
  for (r in 1:nrep) {
    fo <- shuffle_groups(pp, r)[pp]; fe <- matrix(NA, 5, length(lam)); w <- numeric(5)
    for (k in 1:5) { te <- fo == k; f <- glmnet(X[!te, ], yy[!te], family = "binomial", alpha = 0.9, lambda = lam, standardize = FALSE, weights = ww[!te]); pr <- predict(f, X[te, , drop = FALSE], type = "response")
      if (ncol(pr) < length(lam)) pr <- cbind(pr, matrix(pr[, ncol(pr)], nrow(pr), length(lam) - ncol(pr)))
      fe[k, ] <- colSums(((pr > 0.5) != yy[te]) * ww[te]) / sum(ww[te]); w[k] <- sum(ww[te]) }
    cls[r, ] <- colSums(fe * w) / sum(w) }
  cm <- colMeans(cls); list(fit = fit, lam = max(lam[cm <= min(cm)])) }
coxdev <- function(f, X, S) { # Breslow partial log-likelihood per lambda column
  eta <- predict(f, X, type = "link"); apply(eta, 2, function(e) { t <- S[, 1]; d <- S[, 2]; o <- order(-t); e <- e[o]; t <- t[o]; d <- d[o]
    lse <- log(cumsum(exp(e - max(e)))) + max(e); lse <- ave(lse, t, FUN = max)   # risk set = all with time >= t (ties included)
    sum(d * (e - lse)) }) }
cv_cox <- function(X, S, pp, nrep) {   # grouped CV deviance (Verweij & van Houwelingen): ll_full(test+train) - ll_train, at the training-fold fit
  fit <- glmnet(X, S, family = "cox", alpha = 0.9, standardize = FALSE); lam <- fit$lambda; cm <- matrix(NA, nrep, length(lam))
  for (r in 1:nrep) {
    fo <- shuffle_groups(pp, r)[pp]; acc <- numeric(length(lam))
    for (k in 1:5) { te <- fo == k; f <- glmnet(X[!te, ], S[!te, ], family = "cox", alpha = 0.9, lambda = lam, standardize = FALSE)
      a <- coxdev(f, X, S); b <- coxdev(f, X[!te, , drop = FALSE], S[!te, , drop = FALSE]); v <- -2 * (a - b)
      if (length(v) < length(lam)) v <- c(v, rep(v[length(v)], length(lam) - length(v))); acc <- acc + v }
    cm[r, ] <- acc }
  m <- colMeans(cm); list(fit = fit, lam = lam[which.min(m)]) }
fitpred <- function(tr, te, nrep) {   # fit on training rows tr (filtered by use()), predict rows te; returns list(pred, lambda, nnz, train_mean, train_sd)
  X <- feat(tr); tu <- use(tr)
  if (iscox) {
    S <- cbind(time = tt[tu], status = y[tu]); cv <- cv_cox(X[tu, ], S, pat[tu], nrep)
    lp_tr <- predict(cv$fit, X[tu, ], s = cv$lam, type = "link")[, 1]; lp <- predict(cv$fit, X[te, , drop = FALSE], s = cv$lam, type = "link")[, 1]
    mu <- mean(lp_tr); sdv <- sd(lp_tr); if (!is.finite(sdv) || sdv == 0) sdv <- 1
    list(pred = (lp - mu) / sdv, raw = lp, lambda = cv$lam, nnz = sum(as.numeric(as.matrix(coef(cv$fit, s = cv$lam))) != 0), mu = mu, sdv = sdv)
  } else {
    cv <- cv_bin(X[tu, ], y[tu], pat[tu], wts(tu), nrep); pr <- predict(cv$fit, X[te, , drop = FALSE], s = cv$lam, type = "response")[, 1]
    list(pred = pr, raw = pr, lambda = cv$lam, nnz = sum(as.numeric(as.matrix(coef(cv$fit, s = cv$lam)))[-1] != 0), mu = NA, sdv = NA)
  } }
strat_folds <- function(pp, yy, seed, nf) { pl <- tapply(yy, pp, max); Pp <- names(pl)[pl == 1]; Np <- names(pl)[pl == 0]; set.seed(seed); ord <- c(sample(Pp), sample(Np)); setNames(rep(1:nf, length.out = length(ord)), ord) }
fold_of <- strat_folds(pat, y, REP, 10)   # identical to kv_cv.R
if (MODE == "outer") {
  out <- list()
  for (k in 1:10) {
    te <- which(fold_of[pat] == k); tr <- which(fold_of[pat] != k); r <- fitpred(tr, te, 10)
    out[[k]] <- data.frame(Sample = ids[te], Patient = pat[te], y = y[te], cfg = CFG, rep = REP, fold = k, lambda = r$lambda, nnz = r$nnz, n_train = length(use(tr)), mu = r$mu, sdv = r$sdv, raw = r$raw, pred = r$pred)
  }
  write_csv(bind_rows(out), outf(0))
} else {
  for (FOLD in FOLDS[todo]) {
    otr <- which(fold_of[pat] != FOLD); ifo <- strat_folds(pat[otr], y[otr], 1000 * REP + FOLD, 5)[pat[otr]]; out <- list()
    for (j in 1:5) {
      tri <- otr[ifo != j]; tei <- otr[ifo == j]; r <- fitpred(tri, tei, 1)
      out[[j]] <- data.frame(Sample = ids[tei], Patient = pat[tei], y = y[tei], cfg = CFG, rep = REP, fold = FOLD, inner_fold = j, lambda = r$lambda, nnz = r$nnz, raw = r$raw, pred = r$pred)
    }
    write_csv(bind_rows(out), outf(FOLD))
  }
}
w_ <- warnings(); if (length(w_)) print(table(substr(names(w_), 1, 90)))
message("EXE4 FIT DONE ", CFG, " ", MODE, " rep ", REP)
