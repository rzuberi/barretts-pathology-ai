# Killcoyne checks, item 5 (docs/paper_plan_killcoyne_checks.md @ 828ebf7): freeze L-CNV, L-IMG and L-EARLY (their CNV matrix) refitted once on
# all of set C; image block z-scored with set-C statistics; lambda = class-error min of 10 x 5 patient-grouped CV on all of C (set.seed(r));
# alpha 0.9, standardize = FALSE. L-LATE = mean of frozen L-CNV and L-IMG probabilities. Writes aggregates only to OUT (default
# feasibility/paper_plan/killcoyne_mm/checks/frozen); kc_freeze_manifest.py adds extraction settings and hashes.
suppressMessages({library(glmnet); library(readr); library(dplyr); library(BarrettsProgressionRisk)})
K <- "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/feasibility/paper_plan/killcoyne"; M <- paste0(K, "_mm"); OUT <- Sys.getenv("OUT", file.path(M, "checks", "frozen")); dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
C <- read_csv(file.path(M, "set_C.csv"), col_types = cols(.default = "c")); ids <- C$Sample; y <- as.integer(C$Status == "P"); pat <- C$Patient; n <- length(ids)
R <- read_csv(file.path(M, "cnv_their_C.csv"), col_types = cols(sample_id = "c", .default = "d")); CN <- as.matrix(R[, -1]); rownames(CN) <- R$sample_id; CN <- CN[ids, ]
R <- read_csv(file.path(M, "img_mean.csv"), col_types = cols(Sample = "c", .default = "d")); IM <- as.matrix(R[, -1]); rownames(IM) <- R$Sample; IM <- IM[ids, ]
mu <- colMeans(IM); sdv <- apply(IM, 2, sd); IMz <- sweep(sweep(IM, 2, mu), 2, ifelse(sdv > 0, sdv, 1), "/")
shuffle_groups <- function(pp, r) { up <- unique(pp); set.seed(r); sp <- sample(up); setNames(rep(1:5, length.out = length(sp)), sp) }
cvmin <- function(X) {
  fit <- glmnet(X, y, family = "binomial", alpha = 0.9, standardize = FALSE); lam <- fit$lambda; cls <- matrix(NA, 10, length(lam))
  for (r in 1:10) { fo <- shuffle_groups(pat, r)[pat]; fe <- matrix(NA, 5, length(lam)); w <- numeric(5)
    for (k in 1:5) { te <- fo == k; f <- glmnet(X[!te, ], y[!te], family = "binomial", alpha = 0.9, lambda = lam, standardize = FALSE); pr <- predict(f, X[te, , drop = FALSE], type = "response")
      if (ncol(pr) < length(lam)) pr <- cbind(pr, matrix(pr[, ncol(pr)], nrow(pr), length(lam) - ncol(pr)))
      fe[k, ] <- colMeans((pr > 0.5) != y[te]); w[k] <- sum(te) }
    cls[r, ] <- colSums(fe * w) / sum(w) }
  cm <- colMeans(cls); list(fit = fit, lambda = max(lam[cm <= min(cm)])) }
info <- list()
for (nm in c("L_CNV", "L_IMG", "L_EARLY")) {
  X <- switch(nm, L_CNV = CN, L_IMG = IMz, L_EARLY = cbind(CN, IMz)); cv <- cvmin(X); cf <- as.matrix(coef(cv$fit, s = cv$lambda))
  write_csv(tibble(feature = rownames(cf), coefficient = cf[, 1]), file.path(OUT, paste0("coef_", nm, ".csv")))
  p <- predict(cv$fit, X, s = cv$lambda, type = "response")[, 1]; info[[nm]] <- list(lambda = cv$lambda, n_features = ncol(X), nonzero = sum(cf[-1, 1] != 0), insample_mean_prob = mean(p))
  assign(paste0("p_", nm), p)
}
write_csv(tibble(feature = colnames(IM), mean = mu, sd = sdv), file.path(OUT, "image_scaling.csv"))
m <- BarrettsProgressionRisk:::be_model
write_csv(bind_rows(tibble(kind = "tile", feature = names(m$tile.mean), mean = m$tile.mean, sd = m$tile.sd), tibble(kind = "arm", feature = names(m$arms.mean), mean = m$arms.mean, sd = m$arms.sd),
                    tibble(kind = "cx", feature = "cx", mean = m$cx.mean, sd = m$cx.sd)), file.path(OUT, "cnv_scaling_packaged_model.csv"))
stopifnot(identical(colnames(CN), c(names(m$tile.mean), names(m$arms.mean), "cx")))
info$fit <- list(alpha = 0.9, standardize = FALSE, family = "binomial", lambda_rule = "class-error min, 10 x 5 patient-grouped CV on all of set C, set.seed(r) r = 1..10, lambda.min = largest lambda at the minimum",
                 late = "L_LATE probability = (L_CNV probability + L_IMG probability) / 2", training_set = sprintf("set C: %d samples, %d patients, label = sheet Status (P = 1)", n, length(unique(pat))))
info$versions <- list(R = R.version.string, glmnet = as.character(packageVersion("glmnet")), BarrettsProgressionRisk = as.character(packageVersion("BarrettsProgressionRisk")))
info$cnv_preprocessing <- "Features in the column order of coef_L_CNV.csv: 589 5-Mb tiles (hg19 coordinates of the packaged model) minus their overlapping arm(s), 44 arms, cx. Construction as BarrettsProgressionRisk::tileSamples: segmentRawData (QDNAseq 50 kb, gamma2 250) -> tileSegments 5e6 and 'arms' -> unit.var with the packaged model's means/s.d. (cnv_scaling_packaged_model.csv) -> cx = scoreCX on the scaled tiles, then unit.var with cx mean/s.d. -> subtractArms. The training matrix is the packaged model's shipped fit.data (their hg19 processing); features computed on other data are an approximation of that scale."
info$image_preprocessing <- "UNI2 slide embeddings (settings in extraction_settings.json): mean over the sample's tiles (union over the sample's slides), then (x - mean) / sd with image_scaling.csv."
info$insample_check <- list(auroc_L_CNV = NA, note = "in-sample, not a performance estimate")
aucf <- function(yy, s) { r <- rank(s); n1 <- sum(yy); n0 <- length(yy) - n1; (sum(r[yy == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0) }
info$insample_check <- list(auroc_L_CNV = aucf(y, p_L_CNV), auroc_L_IMG = aucf(y, p_L_IMG), auroc_L_EARLY = aucf(y, p_L_EARLY), auroc_L_LATE = aucf(y, (p_L_CNV + p_L_IMG) / 2), note = "in-sample on set C, not a performance estimate")
jsonlite::write_json(info, file.path(OUT, "model_info.json"), auto_unbox = TRUE, pretty = TRUE, digits = NA); message("FREEZE DONE")
