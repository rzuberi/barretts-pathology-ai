# Killcoyne stratified CV, item 4 (docs/paper_plan_killcoyne_cv.md @ 76879b8): freeze L-CNV, L-IMG, L-EARLY on package features, refitted once on
# all of set C with scaling fitted on set C; L-LATE = mean of frozen L-CNV and L-IMG probabilities. Aggregates only.
source("/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/scripts/paper_plan/kv_common.R"); suppressMessages(library(BarrettsProgressionRisk))
OUT <- Sys.getenv("OUT", file.path(M, "cv", "frozen_pkg")); dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
P <- readRDS(file.path(K, "pkg_tiles.rds")); SEG <- P$seg[ids, ]; ARM <- P$arm[ids, ]
loc <- function(nm) { a <- do.call(rbind, strsplit(sub("-", ":", nm), ":")); data.frame(chr = a[, 1], s = as.numeric(a[, 2]), e = as.numeric(a[, 3])) }
lt <- loc(colnames(SEG)); la <- loc(colnames(ARM)); W <- matrix(0, ncol(ARM), ncol(SEG), dimnames = list(colnames(ARM), colnames(SEG))); for (i in seq_len(nrow(la))) W[i, lt$chr == la$chr[i] & lt$s <= la$e[i] & lt$e >= la$s[i]] <- 1
smu <- colMeans(SEG); ssd <- apply(SEG, 2, sd); amu <- colMeans(ARM); asd <- apply(ARM, 2, sd)
sc <- function(X, mu, s) { Z <- sweep(sweep(X, 2, mu), 2, ifelse(s > 0, s, 1), "/"); Z[, s == 0] <- 0; Z }
S <- sc(SEG, smu, ssd); A <- sc(ARM, amu, asd); cxr <- apply(S, 1, function(x) sum(x >= mean(x) + 2 * sd(x) | x <= mean(x) - 2 * sd(x))); cxm <- mean(cxr); cxs <- sd(cxr)
X <- cbind(S - A %*% W, A, cx = (cxr - cxm) / cxs); X[!is.finite(X)] <- 0
ref <- readblock("cnv_pkg_C.csv"); stopifnot(identical(colnames(ref), colnames(X))); recon <- max(abs(ref - X)); message("reconstruction max abs diff vs training block: ", signif(recon, 3))
IM <- load_img(); imu <- colMeans(IM); isd <- apply(IM, 2, sd); IMz <- sc(IM, imu, isd)
info <- list(); ps <- list()
for (nm in c("L_CNV", "L_IMG", "L_EARLY")) {
  Xm <- switch(nm, L_CNV = X, L_IMG = IMz, L_EARLY = cbind(X, IMz)); cv <- cvcurves(Xm, y, pat); cf <- as.matrix(coef(cv$fit, s = cv$class_min))
  write_csv(tibble(feature = rownames(cf), coefficient = cf[, 1]), file.path(OUT, paste0("coef_", nm, ".csv"))); ps[[nm]] <- predict(cv$fit, Xm, s = cv$class_min, type = "response")[, 1]
  info[[nm]] <- list(lambda = cv$class_min, n_features = ncol(Xm), nonzero = sum(cf[-1, 1] != 0))
}
write_csv(tibble(kind = c(rep("tile", length(smu)), rep("arm", length(amu)), "cx"), feature = c(names(smu), names(amu), "cx"), mean = c(smu, amu, cxm), sd = c(ssd, asd, cxs)), file.path(OUT, "cnv_scaling_setC.csv"))
write_csv(as_tibble(W, rownames = "arm"), file.path(OUT, "arm_tile_incidence.csv")); write_csv(tibble(feature = colnames(IM), mean = imu, sd = isd), file.path(OUT, "image_scaling.csv"))
aucf <- function(yy, s) { r <- rank(s); n1 <- sum(yy); n0 <- length(yy) - n1; (sum(r[yy == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0) }
info$fit <- list(alpha = 0.9, standardize = FALSE, family = "binomial", lambda_rule = "class-error min, 10 x 5 patient-grouped CV on all of set C, set.seed(r) r = 1..10; largest lambda at the minimum",
                 late = "L_LATE probability = (L_CNV probability + L_IMG probability) / 2", training_set = sprintf("set C: %d samples, %d patients, label = sheet Status (P = 1)", n, length(unique(pat))))
info$cnv_preprocessing <- paste("hg38 QDNAseq 50 kb raw + fitted counts -> BarrettsProgressionRisk segmentRawData per patient (multipcf, gamma2 250, cutoff 0.008) with the hg38 adaptations of scripts/paper_plan/kr_package.R",
  "(blacklist lifted to hg38, autosomes only, explicit hg38 arms, bundled hg38 chromosome file) -> tileSegments 5e6 (587 hg38 tiles) and 'arms' (44) ->",
  "z-score tiles and arms with cnv_scaling_setC.csv -> cx = count of scaled tiles beyond 2 s.d. of the sample's own tile mean, then z-score with the cx row ->",
  "tile minus sum of overlapping scaled arms (arm_tile_incidence.csv) -> columns in the order of coef_L_CNV.csv; non-finite values set to 0.")
info$image_preprocessing <- "UNI2 (tile 224, level 2) slide embeddings, 256 tiles per slide (extraction_settings.json): mean over the sample's tiles (union over its slides), then z-score with image_scaling.csv."
info$reconstruction_max_abs_diff <- recon
info$insample_check <- list(auroc_L_CNV = aucf(y, ps$L_CNV), auroc_L_IMG = aucf(y, ps$L_IMG), auroc_L_EARLY = aucf(y, ps$L_EARLY), auroc_L_LATE = aucf(y, (ps$L_CNV + ps$L_IMG) / 2), note = "in-sample on set C, not a performance estimate")
info$versions <- list(R = R.version.string, glmnet = as.character(packageVersion("glmnet")), BarrettsProgressionRisk = as.character(packageVersion("BarrettsProgressionRisk")))
jsonlite::write_json(info, file.path(OUT, "model_info.json"), auto_unbox = TRUE, pretty = TRUE, digits = NA); message("KV FREEZE DONE")
