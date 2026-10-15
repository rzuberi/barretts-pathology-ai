# Item 5, scanner transfer (docs/paper_triage_robustness.md @ 32653f1): refits L-IMG only (glmnet alpha 0.9, standardize FALSE, lambda = class-error min of the
# 10 x 5 patient-grouped CV, kv_common.R cvcurves unchanged; image z-scored on training rows). Pre-event samples only.
# JOB = within_<s39|s10>_<rep>  : stratified patient-grouped 10-fold CV (kv_cv.R fold rule) restricted to one scanner, repeat rep.
# JOB = cross_<a|b>_<s39|s10>_<none|perscan> : train scanner = the named one; (a) test = other scanner minus training patients; (b) test = all of the other
#        scanner, train = named scanner minus test patients; plus inner 5-fold OOF on the training rows (hz_fit.R strat_folds seed 1, cvcurves1) for cut-offs.
source("/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/scripts/paper_plan/kv_common.R")
JOB <- Sys.getenv("JOB"); OD <- file.path(M, "triage_robustness", "scanner"); dir.create(OD, recursive = TRUE, showWarnings = FALSE)
outf <- file.path(OD, paste0(JOB, ".csv")); if (file.exists(outf)) quit(save = "no")
SM <- read_csv(file.path(M, "horizons", "samples.csv"), col_types = cols(Sample = "c", Patient = "c", .default = "c")); SM <- SM[match(ids, SM$Sample), ]; stopifnot(all(SM$Sample == ids))
SD <- read_csv(file.path(M, "horizons", "slide_desc.csv"), col_types = cols(Sample = "c", scanner = "c", .default = "d")); scan <- SD$scanner[match(ids, SD$Sample)]
pre <- SM$pre %in% c("True", "TRUE", "1"); stopifnot(sum(pre) == 571); code <- c(s39 = "C13239-01", s10 = "C13210")
IM <- load_img()
IMh <- IM; for (sc in code) { r <- which(pre & scan == sc); mu <- colMeans(IM[r, ]); sdv <- apply(IM[r, ], 2, sd); IMh[scan == sc, ] <- sweep(sweep(IM[scan == sc, , drop = FALSE], 2, mu), 2, ifelse(sdv > 0, sdv, 1), "/") }
cvcurves1 <- function(X, yy, pp) {   # hz_fit.R:19-25 verbatim
  fit <- glmnet(X, yy, family = "binomial", alpha = 0.9, standardize = FALSE); lam <- fit$lambda
  fo <- shuffle_groups(pp, 1)[pp]; fe <- matrix(NA, 5, length(lam)); w <- numeric(5)
  for (k in 1:5) { te <- fo == k; f <- glmnet(X[!te, ], yy[!te], family = "binomial", alpha = 0.9, lambda = lam, standardize = FALSE); pr <- predict(f, X[te, , drop = FALSE], type = "response")
    if (ncol(pr) < length(lam)) pr <- cbind(pr, matrix(pr[, ncol(pr)], nrow(pr), length(lam) - ncol(pr)))
    fe[k, ] <- colMeans((pr > 0.5) != yy[te]); w[k] <- sum(te) }
  cm <- colSums(fe * w) / sum(w); list(fit = fit, class_min = max(lam[cm <= min(cm)])) }
strat_folds <- function(pp, yy, seed, nf) { pl <- tapply(yy, pp, max); Pp <- names(pl)[pl == 1]; Np <- names(pl)[pl == 0]; set.seed(seed); ord <- c(sample(Pp), sample(Np)); setNames(rep(1:nf, length.out = length(ord)), ord) }
fitpred <- function(X, tr, te, one = FALSE) { cv <- if (one) cvcurves1(X[tr, ], y[tr], pat[tr]) else cvcurves(X[tr, ], y[tr], pat[tr]); predict(cv$fit, X[te, , drop = FALSE], s = cv$class_min, type = "response")[, 1] }
nP <- function(r) length(unique(pat[r][y[r] == 1])); nPat <- function(r) length(unique(pat[r]))
p <- strsplit(JOB, "_")[[1]]; out <- list()
if (p[1] == "within") {
  rr <- which(pre & scan == code[[p[2]]]); rep_ <- as.integer(p[3]); fo <- strat_folds(pat[rr], y[rr], rep_, 10)[pat[rr]]
  for (k in 1:10) { te <- rr[fo == k]; tr <- rr[fo != k]; X <- zs(IM, tr)
    out[[k]] <- data.frame(Sample = ids[te], Patient = pat[te], y = y[te], role = "within", rep = rep_, fold = k, pred = fitpred(X, tr, te)) }
} else {
  dsg <- p[2]; trs <- code[[p[3]]]; tes <- setdiff(code, trs); IMs <- if (p[4] == "perscan") IMh else IM
  if (dsg == "a") { tr <- which(pre & scan == trs); te <- which(pre & scan == tes & !(pat %in% pat[tr])) } else { te <- which(pre & scan == tes); tr <- which(pre & scan == trs & !(pat %in% pat[te])) }
  est <- nPat(te) >= 10 && nP(te) >= 2 && nP(tr) >= 3
  if (!est) { write_csv(data.frame(Sample = NA, Patient = NA, y = NA, role = "not_estimable", rep = NA, fold = NA, pred = NA, n_train = length(tr), n_train_patients = nPat(tr), n_train_P_patients = nP(tr), n_test = length(te), n_test_patients = nPat(te), n_test_P_patients = nP(te)), outf); quit(save = "no") }
  X <- zs(IMs, tr); out[[1]] <- data.frame(Sample = ids[te], Patient = pat[te], y = y[te], role = "test", rep = NA, fold = NA, pred = fitpred(X, tr, te))
  ifo <- strat_folds(pat[tr], y[tr], 1, 5)[pat[tr]]
  for (j in 1:5) { tri <- tr[ifo != j]; tei <- tr[ifo == j]; Xi <- zs(IMs, tri); out[[j + 1]] <- data.frame(Sample = ids[tei], Patient = pat[tei], y = y[tei], role = "inner", rep = NA, fold = j, pred = fitpred(Xi, tri, tei, one = TRUE)) }
}
write_csv(bind_rows(out), outf); message("TB SCANNER DONE ", JOB)
