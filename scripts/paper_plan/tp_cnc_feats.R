# Two papers, agent CNC (docs/two_papers_CNC.md): feature blocks from the per-sample outputs of tp_cnc_seg.R (killcoyne_mm/two_papers_CNC/seg/s_*.rds).
# Blocks (rows = the 676 set C samples, column Sample; written to killcoyne_mm/two_papers_CNC/feats/<name>.csv.gz, raw scale: tp_cnc_refit.R z-scales on
# training rows):
#   cbs / ace / ichor : length-weighted segment value over the package's 5-Mb tiles and arms (coordinates = cnv_pkg_C.csv columns), tile minus its arm,
#                       arms, cx (sample-internal count of tiles beyond its own mean +- 2 SD; km_linear.R uses cohort-standardised tiles instead).
#                       cbs value = log2 of the mean ratio to the sample median (floor 1/8); ace = absolute copies clamped to [0, 10]; ichor = logR.
#                       ace adds cellularity and fit error; ichor adds tumour fraction and ploidy.
#   gis               : genome-instability scores (segments >= 10 bins): fraction genome altered and wGII (|log2| > 0.1 for CBS; rounded copies != 2 for
#                       ACE; copies != 2 for ichorCNA), segment count, breakpoint count, LST-like count, max breakpoints on one chromosome (chromothripsis-
#                       like), ACE ploidy (length-weighted copies), ACE cellularity, ichorCNA tumour fraction and ploidy.
#   cin_drews / cin_mac: CINSignatureQuantification activities (Drews 2022, 17; Macintyre 2018, 7) on ACE absolute segments (copies clamped at 0, hg38).
#   pkg_gis           : the stored joint package block (cnv_pkg_C.csv) + gis.
# Columns missing in > 50% of samples (tiles no segment overlaps) are dropped; other missing cells are imputed with training means in tp_cnc_refit.R.
# Also writes tessera_segments.csv.gz (CBS segments >= 10 bins, log2 ratio clamped to [-3, 3]) for tp_cnc_tessera.py, and feats_info.csv (checks).
.libPaths(c("/mnt/scratche/slow/fmlab/zuberi01/envs/cnc_rlib", .libPaths()))
suppressMessages({library(GenomicRanges); library(dplyr); library(readr)})
K <- "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/feasibility/paper_plan/killcoyne"; M <- paste0(K, "_mm"); CD <- file.path(M, "two_papers_CNC"); FD <- file.path(CD, "feats")
dir.create(FD, showWarnings = FALSE)
smp <- read.csv(file.path(K, "kr_samples.csv"), colClasses = "character"); C <- read.csv(file.path(M, "set_C.csv"), colClasses = "character"); smp <- smp[smp$Sample %in% C$Sample, ]
fs <- sprintf("%s/seg/s_%03d.rds", CD, seq_len(nrow(smp))); miss <- sum(!file.exists(fs)); if (miss > 0) stop(miss, " sample outputs missing")
S <- list(); for (f in fs) S <- c(S, readRDS(f)); stopifnot(identical(names(S), smp$Sample)); ids <- C$Sample   # output rows in set C order
REF <- read_csv(file.path(M, "cnv_pkg_C.csv"), col_types = cols(sample_id = "c", .default = "d"), progress = FALSE)
cols <- setdiff(colnames(REF), c("sample_id", "cx")); loc <- do.call(rbind, strsplit(sub("-", ":", cols), ":")); loc <- data.frame(chr = loc[, 1], s = as.numeric(loc[, 2]), e = as.numeric(loc[, 3]))
isarm <- (loc$e - loc$s) > 6e6; TL <- loc[!isarm, ]; AL <- loc[isarm, ]; tn <- cols[!isarm]; an <- cols[isarm]
W <- sapply(seq_len(nrow(TL)), function(i) which(AL$chr == TL$chr[i] & AL$s <= TL$e[i] & AL$e >= TL$s[i])[1])   # covering arm of each tile (NA in centromere gaps)
GT <- GRanges(TL$chr, IRanges(TL$s, TL$e)); GA <- GRanges(AL$chr, IRanges(AL$s, AL$e))
wmean <- function(seg, G) {   # length-weighted mean of seg$value over each interval of G (NA if no overlap)
  gs <- GRanges(seg$chr, IRanges(seg$start, seg$end)); h <- findOverlaps(G, gs); w <- width(pintersect(G[queryHits(h)], gs[subjectHits(h)]))
  num <- tapply(w * seg$value[subjectHits(h)], factor(queryHits(h), levels = seq_along(G)), sum); den <- tapply(w, factor(queryHits(h), levels = seq_along(G)), sum); as.numeric(num / den) }
tile_block <- function(getseg, post = identity) {
  rows <- lapply(ids, function(sm) { g <- getseg(S[[sm]]); if (is.null(g) || nrow(g) == 0) return(rep(NA, length(tn) + length(an) + 1))
    t <- post(wmean(g, GT)); a <- post(wmean(g, GA)); d <- t - a[W]; cx <- sum(abs(t - mean(t, na.rm = TRUE)) >= 2 * sd(t, na.rm = TRUE), na.rm = TRUE); c(d, a, cx) })
  X <- do.call(rbind, rows); colnames(X) <- c(tn, an, "cx"); X }
put <- function(X, name) { dr <- colMeans(is.na(X)) > 0.5; if (any(dr)) { message(name, ": dropped ", sum(dr), " columns missing in > 50% of samples"); info[[paste0(name, "_cols_dropped")]] <<- sum(dr) }; X <- X[, !dr, drop = FALSE]
  D <- data.frame(Sample = ids, X, check.names = FALSE); write_csv(D, file.path(FD, paste0(name, ".csv.gz"))); message(name, " ", nrow(X), " x ", ncol(X), " NA share ", signif(mean(is.na(X)), 3)) }
info <- list()
# ---- checks carried from tp_cnc_seg.R
info$counts_equal_all <- all(sapply(S, function(o) o$check$counts_equal)); info$ace_fitted <- sum(sapply(S, function(o) !is.null(o[["ace"]]))); info$ichor_fitted <- sum(sapply(S, function(o) !is.null(o[["ichor"]])))
# ---- tiled blocks
Xc <- tile_block(function(o) o[["cbs"]], function(v) log2(pmax(v, 1 / 8))); put(Xc, "cbs")
PT <- readRDS(file.path(K, "pkg_tiles.rds")); stopifnot(all(an %in% colnames(PT$arm)), all(tn %in% colnames(PT$seg)))   # sanity: CBS vs stored package tiles, per sample
rho <- function(A, B) sapply(seq_len(nrow(A)), function(i) suppressWarnings(cor(A[i, ], B[i, ], method = "spearman", use = "complete.obs")))
info$cbs_vs_pkg_arm_spearman_median <- median(rho(Xc[, an], PT$arm[ids, an]), na.rm = TRUE); info$cbs_vs_pkg_tile_spearman_median <- median(rho(Xc[, tn] + Xc[, an][, W], PT$seg[ids, tn]), na.rm = TRUE)
Xa <- tile_block(function(o) if (is.null(o[["ace"]])) NULL else transform(o[["ace"]], value = pmin(pmax(value, 0), 10)))
ace_cell <- sapply(ids, function(sm) if (is.null(S[[sm]][["ace_fit"]])) NA else S[[sm]][["ace_fit"]]$cellularity); ace_err <- sapply(ids, function(sm) if (is.null(S[[sm]][["ace_fit"]])) NA else S[[sm]][["ace_fit"]]$rel_error)
put(cbind(Xa, ace_cellularity = ace_cell, ace_rel_error = ace_err), "ace")
Xi <- tile_block(function(o) o[["ichor"]])
ich_tf <- sapply(ids, function(sm) if (is.null(S[[sm]][["ichor_params"]])) NA else S[[sm]][["ichor_params"]]$tumor_fraction); ich_pl <- sapply(ids, function(sm) if (is.null(S[[sm]][["ichor_params"]])) NA else S[[sm]][["ichor_params"]]$ploidy)
put(cbind(Xi, ichor_tf = ich_tf, ichor_ploidy = ich_pl), "ichor")
# ---- genome-instability scores
gstats <- function(g, alt) {   # g: segments (>= 10 bins) with value; alt: altered flag per segment
  if (is.null(g) || nrow(g) == 0) return(rep(NA, 6)); L <- g$end - g$start + 1
  fga <- sum(L[alt]) / sum(L); wg <- mean(sapply(split(seq_len(nrow(g)), g$chr), function(i) sum(L[i][alt[i]]) / sum(L[i])))
  bp <- sapply(split(seq_len(nrow(g)), g$chr), function(i) if (length(i) < 2) 0 else sum(abs(diff(g$value[i])) > 0.1 | diff(alt[i]) != 0))
  lst <- sum(sapply(split(seq_len(nrow(g)), g$chr), function(i) { if (length(i) < 2) return(0); big <- L[i] >= 1e7; ch <- abs(diff(g$value[i])) > 0.1; sum(ch & big[-1] & big[-length(i)]) }))
  c(fga, wg, nrow(g), sum(bp), lst, max(bp)) }
big <- function(g, nb = 10) if (is.null(g)) NULL else g[g$n_bins >= nb, ]
G1 <- t(sapply(ids, function(sm) { g <- big(S[[sm]][["cbs"]]); if (!is.null(g)) g$value <- log2(pmax(g$value, 1 / 8)); gstats(g, if (is.null(g)) NULL else abs(g$value) > 0.1) }))
G2 <- t(sapply(ids, function(sm) { g <- big(S[[sm]][["ace"]]); if (!is.null(g)) g$value <- pmin(pmax(g$value, 0), 10); gstats(g, if (is.null(g)) NULL else round(g$value) != 2) }))
G3 <- t(sapply(ids, function(sm) { g <- big(S[[sm]][["ichor"]], 1); gstats(g, if (is.null(g)) NULL else g$copies != 2) }))
nm <- c("fga", "wgii", "nseg", "nbreak", "lst", "maxbreak_chr"); colnames(G1) <- paste0("cbs_", nm); colnames(G2) <- paste0("ace_", nm); colnames(G3) <- paste0("ichor_", nm)
ace_pl <- sapply(ids, function(sm) { g <- S[[sm]][["ace"]]; if (is.null(g)) return(NA); L <- g$end - g$start + 1; sum(L * pmin(pmax(g$value, 0), 10)) / sum(L) })
GIS <- cbind(G1, G2, G3, ace_ploidy = ace_pl, ace_cellularity = ace_cell, ichor_tf = ich_tf, ichor_ploidy = ich_pl); put(GIS, "gis")
P <- as.matrix(REF[match(ids, REF$sample_id), -1]); put(cbind(P, GIS), "pkg_gis")
# ---- CIN signatures on ACE absolute segments
suppressMessages(library(CINSignatureQuantification))
SD <- bind_rows(lapply(ids, function(sm) { g <- S[[sm]][["ace"]]; if (is.null(g)) return(NULL); data.frame(chromosome = g$chr, start = g$start, end = g$end, segVal = pmax(g$value, 0), sample = sm) }))
for (mth in c("drews", "mac")) {
  q <- tryCatch(quantifyCNSignatures(SD, experimentName = mth, method = mth, build = "hg38"), error = function(e) { message("CINSIG FAIL ", mth, ": ", conditionMessage(e)); NULL })
  if (is.null(q)) { info[[paste0("cin_", mth)]] <- "failed"; next }
  A <- as.matrix(getActivities(q)); X <- A[match(ids, rownames(A)), , drop = FALSE]; rownames(X) <- NULL; info[[paste0("cin_", mth, "_samples")]] <- sum(rownames(A) %in% ids)
  put(X, paste0("cin_", mth))
}
# ---- segments for TESSERA
TS <- bind_rows(lapply(ids, function(sm) { g <- big(S[[sm]][["cbs"]]); data.frame(Tumor_Sample_Barcode = sm, Chromosome = g$chr, Start = g$start, End = g$end, Segment_Mean = pmin(pmax(log2(pmax(g$value, 1 / 8)), -3), 3)) }))
write_csv(TS, file.path(CD, "tessera_segments.csv.gz")); info$tessera_segments_per_sample_median <- median(table(TS$Tumor_Sample_Barcode))
info$cbs_segments_median <- median(sapply(S, function(o) nrow(o[["cbs"]]))); info$cbs_segments_ge10bins_median <- median(sapply(S, function(o) nrow(big(o[["cbs"]]))))
info$ace_cellularity_quantiles <- paste(signif(quantile(ace_cell, c(0, .1, .5, .9, 1), na.rm = TRUE), 3), collapse = " "); info$ichor_tf_quantiles <- paste(signif(quantile(ich_tf, c(0, .1, .5, .9, 1), na.rm = TRUE), 3), collapse = " ")
fl <- unlist(info); write_csv(tibble(key = names(fl), value = as.character(fl)), file.path(CD, "feats_info.csv")); print(fl); message("TP CNC FEATS DONE")
