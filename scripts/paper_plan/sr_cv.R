# Scanner robustness, item 4 (docs/paper_scanner_robustness.md @ abe96e9): one run of L-IMG with the switch extractor's sample representation.
# Code = kv_cv.R (d69de24, cfg 2 "img") and tb_scanner.R (32653f1, within / cross design b, no harmonisation) with only the image matrix swapped
# (load_img overridden to read IMG_FILE). JOB = repro_<rep> (UNI2-h img_mean.csv, D0 check) | cv_<rep> | within_s39_<rep> | cross_b_s39.
# Row-level outputs -> killcoyne_mm/scanner_robustness/cv/.
source("/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/scripts/paper_plan/kv_common.R")
JOB <- Sys.getenv("JOB"); EXT <- Sys.getenv("EXT", "hoptimus0"); SR <- file.path(M, "scanner_robustness"); OD <- file.path(SR, "cv"); dir.create(OD, recursive = TRUE, showWarnings = FALSE)
p <- strsplit(JOB, "_")[[1]]
IMG_FILE <- if (p[1] == "repro") file.path(M, "img_mean.csv") else file.path(SR, sprintf("img_mean_%s.csv", EXT))
tag <- if (p[1] == "repro") "uni2" else EXT; outf <- file.path(OD, sprintf("%s_%s.csv", JOB, tag)); if (file.exists(outf)) quit(save = "no")
load_img <- function() { R <- read_csv(IMG_FILE, col_types = cols(Sample = "c", .default = "d")); X <- as.matrix(R[, -1]); rownames(X) <- R$Sample; X[ids, ] }
out <- list()
if (p[1] %in% c("repro", "cv")) {   # kv_cv.R, cfg 2 (model img, cnvsrc none), verbatim apart from the image file
  rep_ <- as.integer(p[2]); pl <- tapply(y, pat, max); Pp <- names(pl)[pl == 1]; Np <- names(pl)[pl == 0]
  set.seed(rep_); ord <- c(sample(Pp), sample(Np)); fold_of <- setNames(rep(1:10, length.out = length(ord)), ord)
  feat <- make_features("img", "none", "fold")
  for (k in 1:10) {
    te <- fold_of[pat] == k; tr <- which(!te); X <- feat(tr); cv <- cvcurves(X[tr, ], y[tr], pat[tr])
    pr <- predict(cv$fit, X[te, , drop = FALSE], s = cv$class_min, type = "response")[, 1]; b0 <- as.numeric(coef(cv$fit, s = cv$class_min)[1, 1])
    out[[k]] <- data.frame(Sample = ids[te], Patient = pat[te], y = y[te], cfg = 2, model = "img", cnvsrc = "none", rep = rep_, fold = k, lambda = cv$class_min, intercept = b0, train_prev = mean(y[tr]), pred = pr)
  }
} else {   # tb_scanner.R logic
  SM <- read_csv(file.path(M, "horizons", "samples.csv"), col_types = cols(Sample = "c", Patient = "c", .default = "c")); SM <- SM[match(ids, SM$Sample), ]; stopifnot(all(SM$Sample == ids))
  SD <- read_csv(file.path(M, "horizons", "slide_desc.csv"), col_types = cols(Sample = "c", scanner = "c", .default = "d")); scan <- SD$scanner[match(ids, SD$Sample)]
  pre <- SM$pre %in% c("True", "TRUE", "1"); stopifnot(sum(pre) == 571); code <- c(s39 = "C13239-01", s10 = "C13210"); IM <- load_img()
  cvcurves1 <- function(X, yy, pp) {
    fit <- glmnet(X, yy, family = "binomial", alpha = 0.9, standardize = FALSE); lam <- fit$lambda
    fo <- shuffle_groups(pp, 1)[pp]; fe <- matrix(NA, 5, length(lam)); w <- numeric(5)
    for (k in 1:5) { te <- fo == k; f <- glmnet(X[!te, ], yy[!te], family = "binomial", alpha = 0.9, lambda = lam, standardize = FALSE); pr <- predict(f, X[te, , drop = FALSE], type = "response")
      if (ncol(pr) < length(lam)) pr <- cbind(pr, matrix(pr[, ncol(pr)], nrow(pr), length(lam) - ncol(pr)))
      fe[k, ] <- colMeans((pr > 0.5) != yy[te]); w[k] <- sum(te) }
    cm <- colSums(fe * w) / sum(w); list(fit = fit, class_min = max(lam[cm <= min(cm)])) }
  strat_folds <- function(pp, yy, seed, nf) { pl <- tapply(yy, pp, max); Pp <- names(pl)[pl == 1]; Np <- names(pl)[pl == 0]; set.seed(seed); ord <- c(sample(Pp), sample(Np)); setNames(rep(1:nf, length.out = length(ord)), ord) }
  fitpred <- function(X, tr, te, one = FALSE) { cv <- if (one) cvcurves1(X[tr, ], y[tr], pat[tr]) else cvcurves(X[tr, ], y[tr], pat[tr]); predict(cv$fit, X[te, , drop = FALSE], s = cv$class_min, type = "response")[, 1] }
  if (p[1] == "within") {
    rr <- which(pre & scan == code[[p[2]]]); rep_ <- as.integer(p[3]); fo <- strat_folds(pat[rr], y[rr], rep_, 10)[pat[rr]]
    for (k in 1:10) { te <- rr[fo == k]; tr <- rr[fo != k]; X <- zs(IM, tr)
      out[[k]] <- data.frame(Sample = ids[te], Patient = pat[te], y = y[te], role = "within", rep = rep_, fold = k, pred = fitpred(X, tr, te)) }
  } else {
    stopifnot(p[2] == "b"); trs <- code[[p[3]]]; tes <- setdiff(code, trs)
    te <- which(pre & scan == tes); tr <- which(pre & scan == trs & !(pat %in% pat[te]))
    X <- zs(IM, tr); out[[1]] <- data.frame(Sample = ids[te], Patient = pat[te], y = y[te], role = "test", rep = NA, fold = NA, pred = fitpred(X, tr, te))
    ifo <- strat_folds(pat[tr], y[tr], 1, 5)[pat[tr]]
    for (j in 1:5) { tri <- tr[ifo != j]; tei <- tr[ifo == j]; Xi <- zs(IM, tri); out[[j + 1]] <- data.frame(Sample = ids[tei], Patient = pat[tei], y = y[tei], role = "inner", rep = NA, fold = j, pred = fitpred(Xi, tri, tei, one = TRUE)) }
  }
}
write_csv(bind_rows(out), outf); message("SR CV DONE ", JOB, " ", tag)
