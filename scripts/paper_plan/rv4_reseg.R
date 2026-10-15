# Review round 4, Part B, R36 (docs/paper_review_round4_analyses.md @ 45b77bf): the BarrettsProgressionRisk segmentation re-run on our QDNAseq 50 kb
# raw + fitted counts with the kr_package.R pipeline unchanged (package defaults gamma2 250, cutoff 0.008; hg38 adaptations: bundled chrInfo reader,
# explicit hg38 arms for pcf / multipcf, blacklist lifted to hg38, autosomes only; every segmented sample tiled, QC kept separately), varying only
# which samples enter segmentRawData together:
#   MODE=joint PAT: all of the patient's kr_samples.csv samples jointly (multipcf), exactly as stored; also writes the stored pkg_tiles.rds rows of the
#     same samples (pat_%03d_stored.csv) for the reproduction check;
#   MODE=persample PAT: each of the patient's analysed samples on its own (segmentRawData's single-sample branch: pcf with the sample's own sdev);
#   MODE=past PAT: for each endoscopy e with an analysed sample, the patient's kr_samples.csv samples with Endoscopy <= e jointly; the tiles of the
#     samples at e are kept ("past-only" variant; Endoscopy = kr_prep.py's dense rank of the endoscopy year within the patient).
#   MODE=assemble: gathers the per-sample (and past-only, if complete) tiles of the 676 analysed samples and builds the feature block exactly as
#     km_linear.R cnv_block("pkg") (standardised over the 676 analysed samples, arm-subtracted 5-Mb tiles, arms, cx), after checking that the same
#     construction on pkg_tiles.rds returns cnv_pkg_C.csv; writes review_round4/r36_cnv_pkgps_C.csv (and r36_cnv_pkgpast_C.csv) with cnv_pkg_C.csv's
#     columns, plus raw-tile tables (stored joint, per-sample, past) for the feature-difference checks of rv4_refit_metrics.py.
# Row-level outputs -> killcoyne_mm/review_round4/reseg/; stored files are never written.
source("/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/scripts/paper_plan/kv_common.R")   # ids = the 676 analysed samples, pat, K, M, zs, readblock
suppressMessages({library(BarrettsProgressionRisk); library(tidyverse)})
MODE <- Sys.getenv("MODE"); PAT <- as.integer(Sys.getenv("PAT", "-1"))
R4 <- file.path(M, "review_round4"); RS <- file.path(R4, "reseg")
# --- copied verbatim from kr_package.R: package bug workaround (chrInfo) and hg38 arms for copynumber::pcf / multipcf
chrInfo_bundled <- function(chrs = c(1:22, "X", "Y"), prefix = "chr", build = "hg19", file = NULL, verbose = FALSE) {
  f <- system.file("extdata", paste0(build, "_info.txt"), package = "BarrettsProgressionRisk")
  read.table(f, header = TRUE, sep = "\t", stringsAsFactors = FALSE) %>% as_tibble() %>% mutate(chr = sub(prefix, "", chrom)) %>% mutate(chr = factor(chr, levels = chrs, ordered = TRUE))
}
assignInNamespace("chrInfo", chrInfo_bundled, ns = "BarrettsProgressionRisk")
h38 <- chrInfo_bundled(build = "hg38")
arms38 <- function(d) { cen <- setNames(h38$chr.cent, h38$chr)[as.character(d[[1]])]; ifelse(d[[2]] < cen, "p", "q") }
orig_mpcf <- copynumber::multipcf; orig_pcf <- copynumber::pcf
mpcf38 <- function(data, ..., assembly = "hg19") { if (assembly == "hg38") orig_mpcf(data, arms = arms38(data), ..., assembly = "hg19") else orig_mpcf(data, ..., assembly = assembly) }
pcf38 <- function(data, ..., assembly = "hg19") { if (assembly == "hg38") orig_pcf(data, arms = arms38(data), ..., assembly = "hg19") else orig_pcf(data, ..., assembly = assembly) }
assignInNamespace("multipcf", mpcf38, ns = "copynumber"); assignInNamespace("pcf", pcf38, ns = "copynumber")
# --- sample sheet of the stored pipeline (kr_prep.py), same regex-safe ids as kr_package.R
smp <- read_csv(file.path(K, "kr_samples.csv"), col_types = cols(.default = "c")) %>% mutate(gidx = row_number(), sid = sprintf("K%04dQ", gidx), endo = as.integer(Endoscopy))
bl <- read_tsv(file.path(K, "blacklist_hg38.txt"), col_types = "cii")
read_counts <- function(s) {   # kr_package.R: raw and fitted counts of the samples s joined by location, autosomes only
  rd <- NULL; fd <- NULL
  for (j in seq_len(nrow(s))) {
    r <- read_tsv(s$raw[j], col_types = cols(chrom = col_character()), progress = FALSE); f <- read_tsv(s$fit[j], col_types = cols(chrom = col_character()), progress = FALSE)
    colnames(r)[5] <- s$sid[j]; colnames(f)[5] <- s$sid[j]
    if (is.null(rd)) { rd <- r[, 1:5]; fd <- f[, 1:5] } else { rd <- left_join(rd, r[, c(1, 5)], by = "location"); fd <- left_join(fd, f[, c(1, 5)], by = "location") }
  }
  keep <- rd$chrom %in% as.character(1:22); list(rd = rd[keep, ], fd = fd[keep, ])
}
segment_tiles <- function(s) {   # kr_package.R steps on the samples s segmented together: unscaled 5-Mb and arm tiles of every segmented sample + QC
  cnt <- read_counts(s)
  info <- loadSampleInformation(tibble(Sample = s$sid, Endoscopy = s$endo, Pathology = s$Pathology, `P53 IHC` = as.integer(s$`P53 IHC`)))
  obj <- tryCatch(suppressWarnings(segmentRawData(info, cnt$rd, cnt$fd, blacklist = bl, build = "hg38", cache.dir = tempdir(), verbose = FALSE)),
                  error = function(e) { message("SEGMENT FAIL ", paste(s$Sample, collapse = ","), ": ", conditionMessage(e)); NULL })
  if (is.null(obj)) return(NULL)
  allseg <- obj$seg.vals; if (!is.null(obj$failedQC)) allseg <- cbind(allseg, obj$failedQC[, setdiff(colnames(obj$failedQC), colnames(allseg)), drop = FALSE])
  # package edge case absent from the stored joint runs: when no sample of the call passes QC, seg.vals repeats the descriptor columns (grep of an
  # empty pass list), which breaks tileSegments; keep the first occurrence of each descriptor column and the sample columns (values unchanged)
  keepc <- c("chrom", "arm", "start.pos", "end.pos", "n.probes", intersect(s$sid, colnames(allseg))); allseg <- allseg[, match(keepc, colnames(allseg)), drop = FALSE]
  obj2 <- obj; obj2$seg.vals <- allseg; obj2$residuals <- obj$residuals %>% mutate(Pass = TRUE)   # tile every segmented sample; QC kept separately
  segt <- suppressWarnings(tileSegments(obj2, size = 5e6, verbose = FALSE)); armt <- suppressWarnings(tileSegments(obj2, size = "arms", verbose = FALSE))
  qc <- obj$residuals %>% mutate(samplename = as.character(samplename)); sids <- rownames(segt$tiles); stopifnot(all(rownames(armt$tiles) == sids))
  data.frame(Sample = s$Sample[match(sids, s$sid)], n_segmented_together = nrow(s), qc_pass = qc$Pass[match(sids, qc$samplename)], varMAD_median = qc$varMAD_median[match(sids, qc$samplename)],
             segt$tiles, armt$tiles, check.names = FALSE, stringsAsFactors = FALSE)
}
if (MODE %in% c("joint", "persample", "past")) {
  dir.create(file.path(RS, MODE), recursive = TRUE, showWarnings = FALSE); outf <- file.path(RS, MODE, sprintf("pat_%03d.csv", PAT)); if (file.exists(outf)) quit(save = "no")
  s <- smp %>% filter(as.integer(pat_index) == PAT); stopifnot(nrow(s) > 0); t0 <- Sys.time()
  if (MODE == "joint") {
    res <- segment_tiles(s); P <- readRDS(file.path(K, "pkg_tiles.rds")); st <- intersect(s$Sample, rownames(P$seg))
    write_csv(data.frame(Sample = st, qc_pass = P$qc$Pass[match(st, P$qc$Sample)], P$seg[st, , drop = FALSE], P$arm[st, , drop = FALSE], check.names = FALSE), file.path(RS, MODE, sprintf("pat_%03d_stored.csv", PAT)))
  } else if (MODE == "persample") {
    res <- bind_rows(lapply(which(s$Sample %in% ids), function(j) segment_tiles(s[j, ])))
  } else {
    es <- sort(unique(s$endo[s$Sample %in% ids]))
    res <- bind_rows(lapply(es, function(e) { r <- segment_tiles(s[s$endo <= e, ]); if (is.null(r)) NULL else r[r$Sample %in% s$Sample[s$endo == e], ] }))
  }
  stopifnot(!is.null(res), nrow(res) > 0); secs <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  res <- cbind(mode = MODE, pat_index = PAT, secs = secs, res); write_csv(res, outf)
  message("RV4 RESEG DONE ", MODE, " patient ", PAT, " samples ", nrow(res), " qc pass ", sum(res$qc_pass), " secs ", round(secs))
} else if (MODE == "assemble") {
  P <- readRDS(file.path(K, "pkg_tiles.rds")); tn <- colnames(P$seg); an <- colnames(P$arm); loc <- function(nm) { a <- do.call(rbind, strsplit(sub("-", ":", nm), ":")); data.frame(chr = a[, 1], s = as.numeric(a[, 2]), e = as.numeric(a[, 3])) }
  build_block <- function(SEG, ARM) {   # km_linear.R cnv_block("pkg") verbatim (reconciliation plan B construction, standardised over the 676 analysed samples)
    lt <- loc(colnames(SEG)); la <- loc(colnames(ARM)); W <- matrix(0, ncol(ARM), ncol(SEG)); for (i in seq_len(nrow(la))) W[i, lt$chr == la$chr[i] & lt$s <= la$e[i] & lt$e >= la$s[i]] <- 1
    ref <- seq_len(n); S <- zs(SEG[ids, ], ref); A <- zs(ARM[ids, ], ref); cxr <- apply(S, 1, function(x) sum(x >= mean(x) + 2 * sd(x) | x <= mean(x) - 2 * sd(x))); cx <- (cxr - mean(cxr)) / sd(cxr)
    X <- cbind(S - A %*% W, A, cx = cx); X[!is.finite(X)] <- 0; list(X = X, cx_raw = cxr) }
  REF <- readblock("cnv_pkg_C.csv"); J <- build_block(P$seg, P$arm); stopifnot(identical(colnames(REF), colnames(J$X))); recon <- max(abs(REF - J$X)); message("stored block reconstruction max abs diff: ", signif(recon, 3)); stopifnot(recon < 1e-8)
  tiles_table <- function(SEG, ARM, qc_pass, cx_raw) data.frame(Sample = ids, qc_pass = qc_pass, cx_raw = cx_raw, SEG[ids, tn], ARM[ids, an], check.names = FALSE)
  write_csv(tiles_table(P$seg, P$arm, P$qc$Pass[match(ids, P$qc$Sample)], J$cx_raw), file.path(RS, "tiles_joint_stored_analysed.csv"))
  info <- list(stored_block_reconstruction_max_abs_diff = recon, n_analysed = n, n_tiles = length(tn), n_arms = length(an), n_features = ncol(REF))
  for (mode in c("persample", "past")) {
    fs <- list.files(file.path(RS, mode), "^pat_[0-9]+\\.csv$", full.names = TRUE); if (length(fs) == 0) { info[[mode]] <- list(status = "not run"); next }
    D <- bind_rows(lapply(fs, function(f) read_csv(f, col_types = cols(mode = "c", Sample = "c", qc_pass = "l", .default = "d"), progress = FALSE))); D <- D[!duplicated(D$Sample), ]
    miss <- setdiff(ids, D$Sample); if (length(miss) > 0) { message(mode, ": ", length(miss), " analysed samples missing (", length(fs), " patient files); block not written"); info[[mode]] <- list(status = "incomplete", missing = length(miss), patient_files = length(fs)); next }
    stopifnot(all(tn %in% colnames(D)), all(an %in% colnames(D))); D <- D[match(ids, D$Sample), ]
    SEG <- as.matrix(D[, tn]); ARM <- as.matrix(D[, an]); rownames(SEG) <- rownames(ARM) <- ids; B <- build_block(SEG, ARM); stopifnot(identical(colnames(REF), colnames(B$X)))
    tag <- c(persample = "pkgps", past = "pkgpast")[[mode]]; write_csv(as_tibble(B$X) %>% mutate(sample_id = ids, .before = 1), file.path(R4, sprintf("r36_cnv_%s_C.csv", tag)))
    write_csv(tiles_table(SEG, ARM, D$qc_pass, B$cx_raw), file.path(RS, sprintf("tiles_%s_analysed.csv", mode)))
    info[[mode]] <- list(status = "complete", patient_files = length(fs), qc_pass = sum(D$qc_pass), secs_total = sum(tapply(D$secs, D$pat_index, function(v) v[1])), mean_n_segmented_together = mean(D$n_segmented_together))
    message(mode, ": block written, ", nrow(B$X), " x ", ncol(B$X), ", QC pass ", sum(D$qc_pass))
  }
  fl <- unlist(info); write_csv(tibble(key = names(fl), value = as.character(fl)), file.path(RS, "assemble_info.csv")); message("RV4 ASSEMBLE DONE")
} else stop("unknown MODE ", MODE)
