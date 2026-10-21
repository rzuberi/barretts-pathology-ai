# Exploration E1 (docs/paper_explore_E1.md): coefficients of the H&E model (L-IMG), which are not stored. Refits the kv_cv.R outer loop (cfg 02, img)
# and the hz_fit.R inner loop (img) verbatim (kv_common.R sourced unchanged; cvcurves1 / strat_folds copied from hz_fit.R) and writes, per fit, the
# coefficient vector w (z-scale), intercept, lambda and the z-scoring mean / sd of the training rows, then checks the refit predictions against the
# stored ones. Row-level outputs -> killcoyne_mm/explore_E1/coef/ (cluster only); stored predictions are never written.
# MODE=outer REP=1..10 | MODE=inner REP=1..10 FOLD=1..10
source("/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/scripts/paper_plan/kv_common.R")
MODE <- Sys.getenv("MODE"); REP <- as.integer(Sys.getenv("REP", "1")); FOLD <- as.integer(Sys.getenv("FOLD", "0")); stopifnot(MODE %in% c("outer", "inner"))
OD <- file.path(M, "explore_E1", "coef"); dir.create(OD, recursive = TRUE, showWarnings = FALSE)
outf <- if (MODE == "outer") file.path(OD, sprintf("outer_rep_%02d.csv", REP)) else file.path(OD, sprintf("inner_rep_%02d_fold_%02d.csv", REP, FOLD))
if (file.exists(outf)) quit(save = "no")
IMraw <- load_img(); feat <- function(tr) zs(IMraw, tr)
cvcurves1 <- function(X, yy, pp) {
  fit <- glmnet(X, yy, family = "binomial", alpha = 0.9, standardize = FALSE); lam <- fit$lambda
  fo <- shuffle_groups(pp, 1)[pp]; fe <- matrix(NA, 5, length(lam)); w <- numeric(5)
  for (k in 1:5) { te <- fo == k; f <- glmnet(X[!te, ], yy[!te], family = "binomial", alpha = 0.9, lambda = lam, standardize = FALSE); pr <- predict(f, X[te, , drop = FALSE], type = "response")
    if (ncol(pr) < length(lam)) pr <- cbind(pr, matrix(pr[, ncol(pr)], nrow(pr), length(lam) - ncol(pr)))
    fe[k, ] <- colMeans((pr > 0.5) != yy[te]); w[k] <- sum(te) }
  cm <- colSums(fe * w) / sum(w); list(fit = fit, class_min = max(lam[cm <= min(cm)])) }
strat_folds <- function(pp, yy, seed, nf) { pl <- tapply(yy, pp, max); Pp <- names(pl)[pl == 1]; Np <- names(pl)[pl == 0]; set.seed(seed); ord <- c(sample(Pp), sample(Np)); setNames(rep(1:nf, length.out = length(ord)), ord) }
fold_of <- strat_folds(pat, y, REP, 10)
rows <- list(); chk <- list()
emit <- function(fold, ifold, cv, ref) {
  b <- as.numeric(coef(cv$fit, s = cv$class_min)); mu <- colMeans(IMraw[ref, , drop = FALSE]); sdv <- apply(IMraw[ref, , drop = FALSE], 2, sd)
  w <- b[-1]; stopifnot(length(w) == ncol(IMraw)); w[sdv == 0] <- 0
  data.frame(rep = REP, fold = fold, inner_fold = ifold, kind = c("w", "mu", "sd"), lambda = cv$class_min, intercept = b[1], rbind(w, mu, sdv), row.names = NULL) }
if (MODE == "outer") {
  st <- read_csv(file.path(M, "cv", "preds", sprintf("cfg_02_rep_%02d.csv", REP)), col_types = cols(Sample = "c", .default = "d"))
  for (k in 1:10) {
    te <- fold_of[pat] == k; tr <- which(!te); X <- feat(tr); cv <- cvcurves(X[tr, ], y[tr], pat[tr])
    pr <- predict(cv$fit, X[te, , drop = FALSE], s = cv$class_min, type = "response")[, 1]; rows[[k]] <- emit(k, 0L, cv, tr)
    s <- st[match(ids[te], st$Sample), ]; chk[[k]] <- c(max(abs(s$pred - pr)), all(s$fold == k), abs(s$lambda[1] - cv$class_min))
  }
} else {
  st <- read_csv(file.path(M, "horizons", "inner", sprintf("img_rep_%02d_fold_%02d.csv", REP, FOLD)), col_types = cols(Sample = "c", .default = "d"))
  otr <- which(fold_of[pat] != FOLD); ifo <- strat_folds(pat[otr], y[otr], 1000 * REP + FOLD, 5)[pat[otr]]
  for (j in 1:5) {
    tri <- otr[ifo != j]; tei <- otr[ifo == j]; X <- feat(tri); cv <- cvcurves1(X[tri, ], y[tri], pat[tri])
    pr <- predict(cv$fit, X[tei, , drop = FALSE], s = cv$class_min, type = "response")[, 1]; rows[[j]] <- emit(FOLD, j, cv, tri)
    s <- st[match(ids[tei], st$Sample), ]; chk[[j]] <- c(max(abs(s$pred - pr)), all(s$inner_fold == j), abs(s$lambda[1] - cv$class_min))
  }
}
ck <- do.call(rbind, chk); message("max |dpred| ", max(ck[, 1]), " folds ok ", all(ck[, 2] == 1), " max |dlambda| ", max(ck[, 3]))
stopifnot(max(ck[, 1]) < 1e-6, all(ck[, 2] == 1), max(ck[, 3]) < 1e-9)
write_csv(bind_rows(rows), outf); message("EXE1 COEF DONE ", MODE, " rep ", REP, " fold ", FOLD)
