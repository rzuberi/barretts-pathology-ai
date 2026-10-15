# Review round 2, Part B (docs/paper_review_round2_analyses.md @ d681783): refits of the existing linear models on existing features under the
# existing folds. kv_common.R is sourced unchanged; the kv_cv.R outer loop and the hz_fit.R inner loop are reproduced verbatim (cvcurves1 and
# strat_folds copied from hz_fit.R), with only the training rows, the lambda rule, the labels or the image matrix changed as stated per MODE.
# Row-level outputs -> killcoyne_mm/review_round2/<MODE><OUT_TAG>/; stored predictions are never written.
# MODE=f1_outer CFG=cnv_their|cnv_pkg|img REP=1..10: kv_cv.R outer, training rows = pre-event samples of the training patients (PRE_FILTER=0 -> kv_cv.R).
# MODE=f1_inner CFG REP FOLD: hz_fit.R inner (5-fold OOF for the outer training rows) on the pre-event training rows only (PRE_FILTER=0 -> hz_fit.R).
# MODE=f2 REP FOLD: L-IMG with the training patients subsampled to 48 (set.seed(100*REP+FOLD), stratified by status at the fold's training prevalence).
# MODE=f3 DRAW=0..199 MODEL=cnv_their|img: one patient-bootstrap draw (review_round2/f3/draws.csv, written by rv2_refit_metrics.py TASK=f3_draws):
#   the drawn patients' samples replicated by their multiplicity; one stratified 10-fold CV with set.seed(DRAW) on the original patient IDs (copies
#   share the fold and every group); outer predictions and inner 5-fold OOF predictions (inner folds set.seed(1000*DRAW + fold)); lambda re-chosen.
# MODE=f4 CFG=cnv_their|img REP: kv_cv.R outer with lambda at the binomial-deviance minimum of the same 10 x 5 inner curves (class-error lambda kept too).
# MODE=f5_inner REP FOLD: hz_fit.R inner for L-IMG on the H-optimus-0 sample embeddings (IMG_FILE; with img_mean.csv it reproduces hz_fit.R).
# MODE=f6_outer PERM REP | f6_inner PERM REP FOLD: L-IMG outer / inner with the progression labels permuted at the patient level inside each outer
#   training set (set.seed(100000*PERM + 100*REP + fold); the number of progressor patients is kept; folds come from the real labels).
source("/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/scripts/paper_plan/kv_common.R")
MODE <- Sys.getenv("MODE"); CFG <- Sys.getenv("CFG", "img"); REP <- as.integer(Sys.getenv("REP", "1")); FOLD <- as.integer(Sys.getenv("FOLD", "0"))
DRAW <- as.integer(Sys.getenv("DRAW", "0")); PERM <- as.integer(Sys.getenv("PERM", "1")); MODEL <- Sys.getenv("MODEL", "img")
PRE_FILTER <- Sys.getenv("PRE_FILTER", "1") == "1"; IMG_FILE <- Sys.getenv("IMG_FILE", ""); OUT_TAG <- Sys.getenv("OUT_TAG", "")
RR <- file.path(M, "review_round2"); OD <- file.path(RR, paste0(MODE, OUT_TAG)); dir.create(OD, recursive = TRUE, showWarnings = FALSE)
if (IMG_FILE != "") load_img <- function() { R <- read_csv(IMG_FILE, col_types = cols(Sample = "c", .default = "d")); X <- as.matrix(R[, -1]); rownames(X) <- R$Sample; X[ids, ] }
SM <- read_csv(file.path(M, "horizons", "samples.csv"), col_types = cols(Sample = "c", .default = "c")); SM <- SM[match(ids, SM$Sample), ]; stopifnot(all(SM$Sample == ids))
pre <- SM$pre %in% c("True", "TRUE", "1"); stopifnot(sum(pre) == 571)
# --- copied verbatim from hz_fit.R (inner lambda rule and stratified fold assignment)
cvcurves1 <- function(X, yy, pp) {
  fit <- glmnet(X, yy, family = "binomial", alpha = 0.9, standardize = FALSE); lam <- fit$lambda
  fo <- shuffle_groups(pp, 1)[pp]; fe <- matrix(NA, 5, length(lam)); w <- numeric(5)
  for (k in 1:5) { te <- fo == k; f <- glmnet(X[!te, ], yy[!te], family = "binomial", alpha = 0.9, lambda = lam, standardize = FALSE); pr <- predict(f, X[te, , drop = FALSE], type = "response")
    if (ncol(pr) < length(lam)) pr <- cbind(pr, matrix(pr[, ncol(pr)], nrow(pr), length(lam) - ncol(pr)))
    fe[k, ] <- colMeans((pr > 0.5) != yy[te]); w[k] <- sum(te) }
  cm <- colSums(fe * w) / sum(w); list(fit = fit, class_min = max(lam[cm <= min(cm)])) }
strat_folds <- function(pp, yy, seed, nf) { pl <- tapply(yy, pp, max); Pp <- names(pl)[pl == 1]; Np <- names(pl)[pl == 0]; set.seed(seed); ord <- c(sample(Pp), sample(Np)); setNames(rep(1:nf, length.out = length(ord)), ord) }
# --- kv_common.R cvcurves with the binomial deviance read off the same folds and fits (F4); predictions clamped as in km_linear.R / kr_lopo.R
cvcurves_dev <- function(X, yy, pp) {
  fit <- glmnet(X, yy, family = "binomial", alpha = 0.9, standardize = FALSE); lam <- fit$lambda; cls <- dev <- matrix(NA, 10, length(lam))
  for (r in 1:10) {
    fo <- shuffle_groups(pp, r)[pp]; fe <- fd <- matrix(NA, 5, length(lam)); w <- numeric(5)
    for (k in 1:5) { te <- fo == k; f <- glmnet(X[!te, ], yy[!te], family = "binomial", alpha = 0.9, lambda = lam, standardize = FALSE); pr <- predict(f, X[te, , drop = FALSE], type = "response")
      if (ncol(pr) < length(lam)) pr <- cbind(pr, matrix(pr[, ncol(pr)], nrow(pr), length(lam) - ncol(pr)))
      pc <- pmin(pmax(pr, 1e-5), 1 - 1e-5)
      fe[k, ] <- colMeans((pr > 0.5) != yy[te]); fd[k, ] <- colMeans(-2 * (yy[te] * log(pc) + (1 - yy[te]) * log(1 - pc))); w[k] <- sum(te) }
    cls[r, ] <- colSums(fe * w) / sum(w); dev[r, ] <- colSums(fd * w) / sum(w) }
  cm <- colMeans(cls); dm <- colMeans(dev); list(fit = fit, class_min = max(lam[cm <= min(cm)]), dev_min = lam[which.min(dm)]) }
nnz <- function(fit, s) sum(as.numeric(as.matrix(coef(fit, s = s)))[-1] != 0)
cfg_feat <- function(cfg) { mdl <- sub("_.*$", "", cfg); src <- if (grepl("_", cfg)) sub("^[a-z]+_", "", cfg) else "none"; list(mdl = mdl, src = src, feat = make_features(mdl, src, "fold")) }
perm_labels <- function(trp, seed) { pl <- tapply(y, pat, max)[trp]; set.seed(seed); setNames(as.integer(pl[sample(length(pl))]), trp) }   # F6: patient-level permutation, counts kept
fold_of <- strat_folds(pat, y, REP, 10)   # identical to kv_cv.R: set.seed(REP); P shuffled then NP; cyclic
out <- list()
if (MODE %in% c("f1_outer", "f4")) {
  outf <- file.path(OD, sprintf("%s_rep_%02d.csv", CFG, REP)); if (file.exists(outf)) quit(save = "no"); g <- cfg_feat(CFG)
  for (k in 1:10) {
    te <- fold_of[pat] == k; tr <- which(!te & (pre | !PRE_FILTER | MODE == "f4")); X <- g$feat(tr)
    if (MODE == "f4") {
      cv <- cvcurves_dev(X[tr, ], y[tr], pat[tr]); pc <- predict(cv$fit, X[te, , drop = FALSE], s = cv$class_min, type = "response")[, 1]; pd <- predict(cv$fit, X[te, , drop = FALSE], s = cv$dev_min, type = "response")[, 1]
      out[[k]] <- data.frame(Sample = ids[te], Patient = pat[te], y = y[te], model = g$mdl, cnvsrc = g$src, rep = REP, fold = k, lambda_class = cv$class_min, lambda_dev = cv$dev_min,
                             nnz_class = nnz(cv$fit, cv$class_min), nnz_dev = nnz(cv$fit, cv$dev_min), pred_class = pc, pred_dev = pd)
    } else {
      cv <- cvcurves(X[tr, ], y[tr], pat[tr]); pr <- predict(cv$fit, X[te, , drop = FALSE], s = cv$class_min, type = "response")[, 1]; b0 <- as.numeric(coef(cv$fit, s = cv$class_min)[1, 1])
      out[[k]] <- data.frame(Sample = ids[te], Patient = pat[te], y = y[te], model = g$mdl, cnvsrc = g$src, rep = REP, fold = k, lambda = cv$class_min, intercept = b0, train_prev = mean(y[tr]), n_train = length(tr), pred = pr)
    }
  }
} else if (MODE %in% c("f1_inner", "f5_inner")) {
  if (MODE == "f5_inner") { CFG <- "img"; PRE_FILTER <- FALSE; stopifnot(IMG_FILE != "") }
  outf <- file.path(OD, sprintf("%s_rep_%02d_fold_%02d.csv", CFG, REP, FOLD)); if (file.exists(outf)) quit(save = "no"); g <- cfg_feat(CFG)
  otr <- which(fold_of[pat] != FOLD & (pre | !PRE_FILTER)); ifo <- strat_folds(pat[otr], y[otr], 1000 * REP + FOLD, 5)[pat[otr]]
  for (j in 1:5) {
    tri <- otr[ifo != j]; tei <- otr[ifo == j]; X <- g$feat(tri); cv <- cvcurves1(X[tri, ], y[tri], pat[tri])
    pr <- predict(cv$fit, X[tei, , drop = FALSE], s = cv$class_min, type = "response")[, 1]
    out[[j]] <- data.frame(Sample = ids[tei], Patient = pat[tei], y = y[tei], model = g$mdl, cnvsrc = g$src, rep = REP, fold = FOLD, inner_fold = j, lambda = cv$class_min, pred = pr)
  }
} else if (MODE == "f2") {
  outf <- file.path(OD, sprintf("img_rep_%02d_fold_%02d.csv", REP, FOLD)); if (file.exists(outf)) quit(save = "no"); g <- cfg_feat("img")
  te <- fold_of[pat] == FOLD; trp <- names(fold_of)[fold_of != FOLD]; pl <- tapply(y, pat, max)[trp]; Pp <- names(pl)[pl == 1]; Np <- names(pl)[pl == 0]
  nP <- round(48 * length(Pp) / length(trp)); set.seed(100 * REP + FOLD); keep <- c(sample(Pp, nP), sample(Np, 48 - nP)); tr <- which(pat %in% keep); stopifnot(length(unique(pat[tr])) == 48)
  X <- g$feat(tr); cv <- cvcurves(X[tr, ], y[tr], pat[tr]); pr <- predict(cv$fit, X[te, , drop = FALSE], s = cv$class_min, type = "response")[, 1]
  out[[1]] <- data.frame(Sample = ids[te], Patient = pat[te], y = y[te], model = "img", cnvsrc = "none", rep = REP, fold = FOLD, lambda = cv$class_min, n_train_patients = 48, n_train_P_patients = nP,
                         full_train_P_patients = length(Pp), full_train_patients = length(trp), pred = pr)
} else if (MODE == "f3") {
  outf <- file.path(OD, sprintf("%s_draw_%03d_outer.csv", MODEL, DRAW)); outi <- file.path(OD, sprintf("%s_draw_%03d_inner.csv", MODEL, DRAW)); if (file.exists(outf) && file.exists(outi)) quit(save = "no")
  g <- cfg_feat(MODEL); DR <- read_csv(file.path(RR, "f3", "draws.csv"), col_types = cols(draw = "i", Patient = "c", mult = "i")); d <- DR[DR$draw == DRAW, ]; stopifnot(nrow(d) > 0)
  ri <- integer(0); cp <- integer(0)
  for (i in seq_len(nrow(d))) { rows <- which(pat == d$Patient[i]); for (cc in seq_len(d$mult[i])) { ri <- c(ri, rows); cp <- c(cp, rep(cc, length(rows))) } }
  pp <- pat[ri]; yy <- y[ri]; sid <- ids[ri]; fo3 <- strat_folds(pp, yy, DRAW, 10); inn <- list()
  for (k in 1:10) {
    te <- fo3[pp] == k; X <- g$feat(ri[!te])[ri, ]; cv <- cvcurves(X[!te, ], yy[!te], pp[!te]); pr <- predict(cv$fit, X[te, , drop = FALSE], s = cv$class_min, type = "response")[, 1]
    out[[k]] <- data.frame(Sample = sid[te], Patient = pp[te], copy = cp[te], y = yy[te], model = g$mdl, cnvsrc = g$src, draw = DRAW, fold = k, lambda = cv$class_min, pred = pr)
    otr <- which(!te); ifo <- strat_folds(pp[otr], yy[otr], 1000 * DRAW + k, 5)[pp[otr]]
    for (j in 1:5) {
      tri <- otr[ifo != j]; tei <- otr[ifo == j]; Xi <- g$feat(ri[tri])[ri, ]; cv1 <- cvcurves1(Xi[tri, ], yy[tri], pp[tri]); pri <- predict(cv1$fit, Xi[tei, , drop = FALSE], s = cv1$class_min, type = "response")[, 1]
      inn[[length(inn) + 1]] <- data.frame(Sample = sid[tei], Patient = pp[tei], copy = cp[tei], y = yy[tei], model = g$mdl, cnvsrc = g$src, draw = DRAW, fold = k, inner_fold = j, lambda = cv1$class_min, pred = pri)
    }
  }
  write_csv(bind_rows(inn), outi)
} else if (MODE == "f6_outer") {
  outf <- file.path(OD, sprintf("img_perm_%d_rep_%02d.csv", PERM, REP)); if (file.exists(outf)) quit(save = "no"); g <- cfg_feat("img")
  for (k in 1:10) {
    te <- fold_of[pat] == k; tr <- which(!te); yp <- perm_labels(names(fold_of)[fold_of != k], 100000 * PERM + 100 * REP + k); yy <- y; yy[tr] <- yp[pat[tr]]
    X <- g$feat(tr); cv <- cvcurves(X[tr, ], yy[tr], pat[tr]); pr <- predict(cv$fit, X[te, , drop = FALSE], s = cv$class_min, type = "response")[, 1]
    out[[k]] <- data.frame(Sample = ids[te], Patient = pat[te], y = y[te], model = "img", cnvsrc = "none", perm = PERM, rep = REP, fold = k, lambda = cv$class_min, train_prev_perm = mean(yy[tr]), pred = pr)
  }
} else if (MODE == "f6_inner") {
  outf <- file.path(OD, sprintf("img_perm_%d_rep_%02d_fold_%02d.csv", PERM, REP, FOLD)); if (file.exists(outf)) quit(save = "no"); g <- cfg_feat("img")
  otr <- which(fold_of[pat] != FOLD); yp <- perm_labels(names(fold_of)[fold_of != FOLD], 100000 * PERM + 100 * REP + FOLD); yy <- y; yy[otr] <- yp[pat[otr]]
  ifo <- strat_folds(pat[otr], y[otr], 1000 * REP + FOLD, 5)[pat[otr]]   # real labels: inner folds identical to the stored ones
  for (j in 1:5) {
    tri <- otr[ifo != j]; tei <- otr[ifo == j]; X <- g$feat(tri); cv <- cvcurves1(X[tri, ], yy[tri], pat[tri]); pr <- predict(cv$fit, X[tei, , drop = FALSE], s = cv$class_min, type = "response")[, 1]
    out[[j]] <- data.frame(Sample = ids[tei], Patient = pat[tei], y = y[tei], y_perm = yy[tei], model = "img", cnvsrc = "none", perm = PERM, rep = REP, fold = FOLD, inner_fold = j, lambda = cv$class_min, pred = pr)
  }
} else stop("unknown MODE ", MODE)
write_csv(bind_rows(out), outf); message("RV2 REFIT DONE ", MODE, " ", CFG, " ", MODEL, " rep ", REP, " fold ", FOLD, " draw ", DRAW, " perm ", PERM)
