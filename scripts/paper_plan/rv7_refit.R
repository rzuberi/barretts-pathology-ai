# Review round 7 refits (docs/paper_review_round7_analyses.md @ 724ff97; R43, R52, R54 and the no-change reproduction). Copies rv4_refit.R: kv_common.R is
# sourced unchanged; the kv_cv.R outer loop and the hz_fit.R inner loop (cvcurves1, strat_folds) are reproduced verbatim. Only the analysed set changes,
# as written by rv7_round7.py TASK=prep to killcoyne_mm/review_round7/sets/<VAR>.csv (Sample, Patient, y; existing rows in set C order, added rows after).
# Folds: existing patients keep the fold they have in set C (outer: strat_folds(set C, REP, 10); inner: strat_folds(set C outer training patients,
# 1000 * REP + FOLD, 5)); added patients are assigned, in sorted patient order, to the fold with the fewest patients of their class (variant labels),
# ties to the lowest fold index, separately in each repeat (and each inner split). Row-level outputs -> killcoyne_mm/review_round7/refit/<VAR>/;
# stored predictions are never written.
# VAR=base|r43a|r43b|r43c|r52a|r52b|r54 MODE=outer|inner CFG=cnv_their|img REP=1..10 [FOLD=1..10; FOLD=0 (default) = all ten inner splits in turn]
source("/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/scripts/paper_plan/kv_common.R")
VAR <- Sys.getenv("VAR"); MODE <- Sys.getenv("MODE"); CFG <- Sys.getenv("CFG"); REP <- as.integer(Sys.getenv("REP", "1")); FOLDS <- as.integer(Sys.getenv("FOLD", "0"))
stopifnot(CFG %in% c("cnv_their", "img"), MODE %in% c("outer", "inner"))
R7 <- file.path(M, "review_round7"); OD <- file.path(R7, "refit", VAR); dir.create(OD, recursive = TRUE, showWarnings = FALSE)
if (FOLDS == 0) FOLDS <- 1:10
outf <- function(fold) if (MODE == "outer") file.path(OD, sprintf("outer_%s_rep_%02d.csv", CFG, REP)) else file.path(OD, sprintf("inner_%s_rep_%02d_fold_%02d.csv", CFG, REP, fold))
todo <- if (MODE == "outer") file.exists(outf(0)) == FALSE else !file.exists(sapply(FOLDS, outf))
if (!any(todo)) quit(save = "no")
# ---- set C as stored (kv_common.R globals) and its folds
ids0 <- ids; y0 <- y; pat0 <- pat
strat_folds <- function(pp, yy, seed, nf) { pl <- tapply(yy, pp, max); Pp <- names(pl)[pl == 1]; Np <- names(pl)[pl == 0]; set.seed(seed); ord <- c(sample(Pp), sample(Np)); setNames(rep(1:nf, length.out = length(ord)), ord) }
fold0 <- strat_folds(pat0, y0, REP, 10)   # identical to kv_cv.R: set.seed(REP); P shuffled then NP; cyclic
# ---- the variant set and its feature rows (stored blocks for existing samples; review_round7/sets/*_added.csv for added samples)
S <- read_csv(file.path(R7, "sets", paste0(VAR, ".csv")), col_types = cols(Sample = "c", Patient = "c", y = "i", .default = "c"))
stopifnot(!anyDuplicated(S$Sample), all(S$y %in% 0:1))
added <- setdiff(S$Sample, ids0)
if (CFG == "img") {
  RAW <- load_img()
  if (length(added)) { A <- read_csv(file.path(R7, "sets", "img_added.csv"), col_types = cols(Sample = "c", .default = "d")); XA <- as.matrix(A[, -1]); rownames(XA) <- A$Sample; stopifnot(identical(colnames(XA), colnames(RAW))); RAW <- rbind(RAW, XA[added, , drop = FALSE]) }
} else {
  RAW <- readblock("cnv_their_C.csv")
  if (length(added)) { A <- read_csv(file.path(R7, "sets", "cnv_their_added.csv"), col_types = cols(sample_id = "c", .default = "d")); XA <- as.matrix(A[, -1]); rownames(XA) <- A$sample_id; stopifnot(identical(colnames(XA), colnames(RAW))); RAW <- rbind(RAW, XA[added, , drop = FALSE]) }
}
ids <- S$Sample; y <- S$y; pat <- S$Patient; n <- length(ids); RAW <- RAW[ids, , drop = FALSE]; stopifnot(all(is.finite(RAW)))
if (CFG == "img") { IMraw <- RAW; feat <- function(tr) zs(IMraw, tr) } else { CN <- RAW; feat <- function(tr) CN }
mdl <- sub("_.*$", "", CFG); src <- if (CFG == "img") "none" else "their"
plab <- tapply(y, pat, max)   # variant patient labels
assign_added <- function(fo, pats, nf) {   # fo: named folds of existing patients; pats: all patients needing a fold
  new <- sort(setdiff(pats, names(fo)))
  for (p in new) { have <- intersect(names(fo), pats); cnt <- sapply(1:nf, function(k) sum(fo[have] == k & plab[have] == plab[[p]])); fo[p] <- which.min(cnt) }
  fo[pats] }
up <- unique(pat); fold_of <- assign_added(fold0[intersect(names(fold0), up)], up, 10)
# ---- cvcurves1 copied verbatim from hz_fit.R (inner lambda rule)
cvcurves1 <- function(X, yy, pp) {
  fit <- glmnet(X, yy, family = "binomial", alpha = 0.9, standardize = FALSE); lam <- fit$lambda
  fo <- shuffle_groups(pp, 1)[pp]; fe <- matrix(NA, 5, length(lam)); w <- numeric(5)
  for (k in 1:5) { te <- fo == k; f <- glmnet(X[!te, ], yy[!te], family = "binomial", alpha = 0.9, lambda = lam, standardize = FALSE); pr <- predict(f, X[te, , drop = FALSE], type = "response")
    if (ncol(pr) < length(lam)) pr <- cbind(pr, matrix(pr[, ncol(pr)], nrow(pr), length(lam) - ncol(pr)))
    fe[k, ] <- colMeans((pr > 0.5) != yy[te]); w[k] <- sum(te) }
  cm <- colSums(fe * w) / sum(w); list(fit = fit, class_min = max(lam[cm <= min(cm)])) }
if (MODE == "outer") {
  out <- list()
  for (k in 1:10) {
    te <- fold_of[pat] == k; tr <- which(!te); X <- feat(tr); cv <- cvcurves(X[tr, ], y[tr], pat[tr])
    pr <- predict(cv$fit, X[te, , drop = FALSE], s = cv$class_min, type = "response")[, 1]; b0 <- as.numeric(coef(cv$fit, s = cv$class_min)[1, 1])
    out[[k]] <- data.frame(Sample = ids[te], Patient = pat[te], y = y[te], var = VAR, model = mdl, cnvsrc = src, rep = REP, fold = k, lambda = cv$class_min, intercept = b0, train_prev = mean(y[tr]), pred = pr)
  }
  write_csv(bind_rows(out), outf(0))
} else {
  for (FOLD in FOLDS[todo]) {
    otr0 <- which(fold0[pat0] != FOLD); ifo0 <- strat_folds(pat0[otr0], y0[otr0], 1000 * REP + FOLD, 5)
    otr <- which(fold_of[pat] != FOLD); upi <- unique(pat[otr]); ifo <- assign_added(ifo0[intersect(names(ifo0), upi)], upi, 5)[pat[otr]]
    out <- list()
    for (j in 1:5) {
      tri <- otr[ifo != j]; tei <- otr[ifo == j]; X <- feat(tri); cv <- cvcurves1(X[tri, ], y[tri], pat[tri])
      pr <- predict(cv$fit, X[tei, , drop = FALSE], s = cv$class_min, type = "response")[, 1]
      out[[j]] <- data.frame(Sample = ids[tei], Patient = pat[tei], y = y[tei], var = VAR, model = mdl, cnvsrc = src, rep = REP, fold = FOLD, inner_fold = j, lambda = cv$class_min, pred = pr)
    }
    write_csv(bind_rows(out), outf(FOLD))
  }
}
message("RV7 REFIT DONE ", VAR, " ", MODE, " ", CFG, " rep ", REP)
