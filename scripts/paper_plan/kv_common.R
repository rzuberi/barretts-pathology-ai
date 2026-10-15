# Shared pieces for kv_cv.R / kv_lopo_intercepts.R / kv_freeze.R (docs/paper_plan_killcoyne_cv.md @ 76879b8): set C, blocks, glmnet CV.
suppressMessages({library(glmnet); library(readr); library(dplyr)})
K <- "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/feasibility/paper_plan/killcoyne"; M <- paste0(K, "_mm")
C <- read_csv(file.path(M, "set_C.csv"), col_types = cols(.default = "c")); ids <- C$Sample; y <- as.integer(C$Status == "P"); pat <- C$Patient; n <- length(ids)
zs <- function(Mx, ref) { mu <- colMeans(Mx[ref, , drop = FALSE]); sdv <- apply(Mx[ref, , drop = FALSE], 2, sd); Z <- sweep(sweep(Mx, 2, mu), 2, ifelse(sdv > 0, sdv, 1), "/"); Z[, sdv == 0] <- 0; Z }
readblock <- function(f) { R <- read_csv(file.path(M, f), col_types = cols(sample_id = "c", .default = "d")); X <- as.matrix(R[, -1]); rownames(X) <- R$sample_id; X[ids, , drop = FALSE] }
load_img <- function() { R <- read_csv(file.path(M, "img_mean.csv"), col_types = cols(Sample = "c", .default = "d")); X <- as.matrix(R[, -1]); rownames(X) <- R$Sample; X[ids, ] }
gr <- c(NDBE = 0, ID = 1, LGD = 2, HGD = 3, IMC = 4)[C$Pathology]; GR <- matrix((gr - mean(gr)) / sd(gr), ncol = 1, dimnames = list(ids, "grade"))
shuffle_groups <- function(pp, r) { up <- unique(pp); set.seed(r); sp <- sample(up); setNames(rep(1:5, length.out = length(sp)), sp) }
cvcurves <- function(X, yy, pp) {
  fit <- glmnet(X, yy, family = "binomial", alpha = 0.9, standardize = FALSE); lam <- fit$lambda; cls <- matrix(NA, 10, length(lam))
  for (r in 1:10) {
    fo <- shuffle_groups(pp, r)[pp]; fe <- matrix(NA, 5, length(lam)); w <- numeric(5)
    for (k in 1:5) { te <- fo == k; f <- glmnet(X[!te, ], yy[!te], family = "binomial", alpha = 0.9, lambda = lam, standardize = FALSE); pr <- predict(f, X[te, , drop = FALSE], type = "response")
      if (ncol(pr) < length(lam)) pr <- cbind(pr, matrix(pr[, ncol(pr)], nrow(pr), length(lam) - ncol(pr)))
      fe[k, ] <- colMeans((pr > 0.5) != yy[te]); w[k] <- sum(te) }
    cls[r, ] <- colSums(fe * w) / sum(w) }
  cm <- colMeans(cls); list(fit = fit, class_min = max(lam[cm <= min(cm)])) }
# model -> feature builder given training rows (image block always fold-honest here)
make_features <- function(model, cnvsrc, img_scaling = "fold") {
  CN <- if (cnvsrc != "none") readblock(sprintf("cnv_%s_C.csv", cnvsrc)) else NULL; IMraw <- if (grepl("img|early", model)) load_img() else NULL
  IMs <- function(tr) if (img_scaling == "fold") zs(IMraw, tr) else zs(IMraw, seq_len(n))
  function(tr) switch(model, cnv = CN, img = IMs(tr), early = cbind(CN, IMs(tr)), grade = cbind(GR, pad = 0), cnvgrade = cbind(CN, GR), earlygrade = cbind(CN, IMs(tr), GR))
}
