# Two papers, agent P2N (docs/two_papers_P2N.md): L-IMG and L-CNV refitted under a patient-level label permutation (selection-aware null, full-refit version).
# kv_common.R is sourced unchanged (set C, zs, readblock, load_img, cvcurves = the kv_cv.R outer lambda rule). The kv_cv.R outer loop and the hz_fit.R inner loop
# (cvcurves1, strat_folds copied verbatim, as rv4_refit.R / ex_e3_refit.R) are reproduced; the ONLY change is the label vector: y is replaced by the permuted patient label
# of shuffle SHUF (killcoyne_mm/two_papers_P2N/perms.csv; SHUF = 0 is the true labels and must reproduce the stored predictions exactly). Because the outer and inner fold
# assignments are stratified by progressor status (strat_folds), they are re-derived from the permuted labels, exactly as the original pipeline would have done.
# MODEL = img (cfg_02 / horizons inner img_*) | cnv (cfg_00, shipped block cnv_their_C.csv / horizons inner cnv_their_*); REPS = "a-b" (repeat seeds).
# IMGFEAT=clean (22 Oct): L-IMG on agent ART's clean-tile features instead of img_mean.csv (files imgclean_*).
# Row-level outputs (cluster only): killcoyne_mm/two_papers_P2N/refit/sXXXX/<MODEL><IMGFEAT>_{outer,inner}_rep_XX.csv.gz
source("/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/scripts/paper_plan/kv_common.R")
SHUF <- as.integer(Sys.getenv("SHUF")); MODEL <- Sys.getenv("MODEL"); RR <- as.integer(strsplit(Sys.getenv("REPS", "1-10"), "-")[[1]]); REPSV <- RR[1]:RR[length(RR)]
stopifnot(MODEL %in% c("img", "cnv"))
XD <- file.path(M, "two_papers_P2N"); OD <- file.path(XD, "refit", sprintf("s%04d", SHUF)); dir.create(OD, recursive = TRUE, showWarnings = FALSE)
if (SHUF > 0) {
  PM <- read_csv(file.path(XD, "perms.csv"), col_types = cols(.default = "i", Patient = "c")); col <- sprintf("s%04d", SHUF); stopifnot(col %in% names(PM))
  lab <- setNames(PM[[col]], PM$Patient); stopifnot(all(pat %in% names(lab))); y <- as.integer(lab[pat])
}
cvcurves1 <- function(X, yy, pp) {   # verbatim hz_fit.R
  fit <- glmnet(X, yy, family = "binomial", alpha = 0.9, standardize = FALSE); lam <- fit$lambda
  fo <- shuffle_groups(pp, 1)[pp]; fe <- matrix(NA, 5, length(lam)); w <- numeric(5)
  for (k in 1:5) { te <- fo == k; f <- glmnet(X[!te, ], yy[!te], family = "binomial", alpha = 0.9, lambda = lam, standardize = FALSE); pr <- predict(f, X[te, , drop = FALSE], type = "response")
    if (ncol(pr) < length(lam)) pr <- cbind(pr, matrix(pr[, ncol(pr)], nrow(pr), length(lam) - ncol(pr)))
    fe[k, ] <- colMeans((pr > 0.5) != yy[te]); w[k] <- sum(te) }
  cm <- colSums(fe * w) / sum(w); list(fit = fit, class_min = max(lam[cm <= min(cm)])) }
strat_folds <- function(pp, yy, seed, nf) { pl <- tapply(yy, pp, max); Pp <- names(pl)[pl == 1]; Np <- names(pl)[pl == 0]; set.seed(seed); ord <- c(sample(Pp), sample(Np)); setNames(rep(1:nf, length.out = length(ord)), ord) }
IMGFEAT <- Sys.getenv("IMGFEAT", "")   # "" = stored img_mean.csv; "clean" = agent ART's clean-tile mean embedding (killcoyne_mm/two_papers_ART/feats/clean.csv.gz, as tp_art_refit.R reads it)
rdart <- function(f) { R <- read_csv(f, col_types = cols(Sample = "c", .default = "d"), progress = FALSE); X <- as.matrix(R[, -1]); rownames(X) <- R$Sample; stopifnot(all(ids %in% rownames(X))); X <- X[ids, , drop = FALSE]; stopifnot(!anyNA(X)); X }
if (MODEL == "img") { IMraw <- if (nzchar(IMGFEAT)) rdart(file.path(M, "two_papers_ART", "feats", paste0(IMGFEAT, ".csv.gz"))) else load_img(); feat <- function(tr) zs(IMraw, tr) } else { CN <- readblock("cnv_their_C.csv"); feat <- function(tr) CN }   # kv_common make_features("img"/"cnv","their"); with no missing values zs equals tp_art_refit.R scale_tr
MTAG <- paste0(MODEL, IMGFEAT)
for (REP in REPSV) {
  fo_ <- file.path(OD, sprintf("%s_outer_rep_%02d.csv.gz", MTAG, REP)); fi_ <- file.path(OD, sprintf("%s_inner_rep_%02d.csv.gz", MTAG, REP))
  if (file.exists(fo_) && file.exists(fi_)) next
  fold_of <- strat_folds(pat, y, REP, 10); outO <- list(); outI <- list()
  for (k in 1:10) {
    te <- fold_of[pat] == k; tr <- which(!te); X <- feat(tr); cv <- cvcurves(X[tr, ], y[tr], pat[tr])
    outO[[k]] <- data.frame(Sample = ids[te], rep = REP, fold = k, pred = predict(cv$fit, X[te, , drop = FALSE], s = cv$class_min, type = "response")[, 1])
    otr <- which(fold_of[pat] != k); ifo <- strat_folds(pat[otr], y[otr], 1000 * REP + k, 5)[pat[otr]]
    for (j in 1:5) {
      tri <- otr[ifo != j]; tei <- otr[ifo == j]; X <- feat(tri); cv <- cvcurves1(X[tri, ], y[tri], pat[tri])
      outI[[length(outI) + 1]] <- data.frame(Sample = ids[tei], rep = REP, fold = k, inner_fold = j, pred = predict(cv$fit, X[tei, , drop = FALSE], s = cv$class_min, type = "response")[, 1])
    }
  }
  write_csv(bind_rows(outI), paste0(fi_, ".tmp.gz")); write_csv(bind_rows(outO), paste0(fo_, ".tmp.gz"))
  file.rename(paste0(fi_, ".tmp.gz"), fi_); file.rename(paste0(fo_, ".tmp.gz"), fo_)
}
message("TPP2N REFIT DONE s", SHUF, " ", MTAG, " reps ", paste(range(REPSV), collapse = "-"))
