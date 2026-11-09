# Two papers, agent CND (docs/two_papers_CND.md), Analysis B: binomial thinning of the raw 50-kb bin counts, then the stored package feature pipeline.
# MODE=pat TGT=<x> PAT=<n> [FORCE=1]: for every kr_samples.csv sample of patient n (the samples the stored pipeline segments together):
#   native depth = total.reads x 50 / 3,088,269,832 (P1B depth_total); keep fraction p = TGT / native depth; if p >= 1 the sample is left as is
#   (stored raw and fitted files used). Otherwise counts ~ Binomial(stored count, p) per bin, seed = gidx + 100000 x round(1000 TGT); the QDNAseq
#   GC/mappability fit is re-estimated on the thinned counts with the stored run's call (applyFilters(residual, blacklist) -> estimateCorrection,
#   which reproduces the stored fit to 5e-13), rounded to 3 decimals as in the stored 50.fitted_read_counts.txt. Then rv4_reseg.R segment_tiles
#   (joint multipcf of the patient's samples, 5-Mb + arm tiles, QC kept separately), unchanged. FORCE=1 re-estimates the fit even when p >= 1
#   (native-depth reproduction check against the stored tiles).
# MODE=assemble TGT=<x>: build the feature block exactly as rv4_reseg.R / km_linear.R cnv_block("pkg") over the 676 analysed samples.
# Row-level output -> killcoyne_mm/two_papers_CND/thin/<tag>/ (cluster only). Source data read-only.
suppressMessages({library(QDNAseq); library(Biobase)})
TGT <- as.numeric(Sys.getenv("TGT")); FORCE <- Sys.getenv("FORCE", "0") == "1"; MODE_CND <- Sys.getenv("MODE"); PAT <- as.integer(Sys.getenv("PAT", "-1"))
Sys.setenv(MODE = "none")   # rv4_reseg.R: definitions only (its dispatch ends in stop("unknown MODE"))
tryCatch(source("/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/scripts/paper_plan/rv4_reseg.R"), error = function(e) if (!grepl("unknown MODE", conditionMessage(e))) stop(e))
TAG <- if (FORCE) "nat_force" else sprintf("thin_%03d", round(TGT * 100)); OD <- file.path(M, "two_papers_CND", "thin", TAG); dir.create(OD, recursive = TRUE, showWarnings = FALSE)
GLEN <- 3088269832
if (MODE_CND == "pat") {
  outf <- file.path(OD, sprintf("pat_%03d.csv", PAT)); if (file.exists(outf)) quit(save = "no")
  s <- smp %>% filter(as.integer(pat_index) == PAT); stopifnot(nrow(s) > 0); td <- tempfile("cnd_"); dir.create(td); t0 <- Sys.time(); info <- list()
  for (j in seq_len(nrow(s))) {
    rc <- readRDS(file.path(dirname(s$raw[j]), "50.read_counts.rds")); cn <- assayDataElement(rc, "counts")[, 1]; tot <- sum(cn); dep <- tot * 50 / GLEN; p <- TGT / dep
    if (p < 1 || FORCE) {
      if (p < 1) { set.seed(s$gidx[j] + 100000L * round(1000 * TGT)); cn <- rbinom(length(cn), cn, p); assayDataElement(rc, "counts")[, 1] <- cn }
      fit <- assayDataElement(estimateCorrection(applyFilters(rc, residual = TRUE, blacklist = TRUE)), "fit")[, 1]
      fd <- fData(rc); loc <- tibble(location = featureNames(rc), chrom = as.character(fd$chromosome), start = fd$start, end = fd$end)
      ref <- read_tsv(s$raw[j], col_types = cols(chrom = col_character()), progress = FALSE); stopifnot(identical(ref$location, loc$location), nrow(ref) == length(cn))
      rf <- file.path(td, paste0(j, ".raw.txt")); ff <- file.path(td, paste0(j, ".fit.txt"))
      write_tsv(bind_cols(loc, setNames(tibble(as.integer(cn)), s$Sample[j])), rf); write_tsv(bind_cols(loc, setNames(tibble(round(fit, 3)), s$Sample[j])), ff)
      s$raw[j] <- rf; s$fit[j] <- ff
    }
    info[[j]] <- data.frame(Sample = s$Sample[j], depth_native = dep, keep_p = min(p, 1), thinned = p < 1, depth_after = sum(cn) * 50 / GLEN)
  }
  res <- segment_tiles(s); unlink(td, recursive = TRUE); stopifnot(!is.null(res), nrow(res) > 0)
  res <- cbind(tag = TAG, pat_index = PAT, secs = as.numeric(difftime(Sys.time(), t0, units = "secs")), left_join(res, bind_rows(info), by = "Sample")); write_csv(res, outf)
  message("TP CND THIN DONE ", TAG, " patient ", PAT, " samples ", nrow(res), " thinned ", sum(res$thinned), " qc pass ", sum(res$qc_pass))
} else if (MODE_CND == "assemble") {
  P <- readRDS(file.path(K, "pkg_tiles.rds")); tn <- colnames(P$seg); an <- colnames(P$arm); loc <- function(nm) { a <- do.call(rbind, strsplit(sub("-", ":", nm), ":")); data.frame(chr = a[, 1], s = as.numeric(a[, 2]), e = as.numeric(a[, 3])) }
  build_block <- function(SEG, ARM) {   # rv4_reseg.R build_block (= km_linear.R cnv_block("pkg")) verbatim
    lt <- loc(colnames(SEG)); la <- loc(colnames(ARM)); W <- matrix(0, ncol(ARM), ncol(SEG)); for (i in seq_len(nrow(la))) W[i, lt$chr == la$chr[i] & lt$s <= la$e[i] & lt$e >= la$s[i]] <- 1
    ref <- seq_len(n); S <- zs(SEG[ids, ], ref); A <- zs(ARM[ids, ], ref); cxr <- apply(S, 1, function(x) sum(x >= mean(x) + 2 * sd(x) | x <= mean(x) - 2 * sd(x))); cx <- (cxr - mean(cxr)) / sd(cxr)
    X <- cbind(S - A %*% W, A, cx = cx); X[!is.finite(X)] <- 0; list(X = X, cx_raw = cxr) }
  REF <- readblock("cnv_pkg_C.csv")
  fs <- list.files(OD, "^pat_[0-9]+\\.csv$", full.names = TRUE); D <- bind_rows(lapply(fs, function(f) read_csv(f, col_types = cols(tag = "c", Sample = "c", qc_pass = "l", thinned = "l", .default = "d"), progress = FALSE)))
  miss <- setdiff(ids, D$Sample); if (length(miss) > 0) stop(length(miss), " analysed samples missing from ", length(fs), " patient files")
  D <- D[match(ids, D$Sample), ]; SEG <- as.matrix(D[, tn]); ARM <- as.matrix(D[, an]); rownames(SEG) <- rownames(ARM) <- ids
  B <- build_block(SEG, ARM); stopifnot(identical(colnames(REF), colnames(B$X)))
  write_csv(as_tibble(B$X) %>% mutate(sample_id = ids, .before = 1), file.path(OD, "cnv_block_C.csv"))
  write_csv(data.frame(Sample = ids, D[, c("depth_native", "keep_p", "thinned", "depth_after", "qc_pass", "varMAD_median", "n_segmented_together")], cx_raw = B$cx_raw), file.path(OD, "samples_C.csv"))
  message("TP CND ASSEMBLE DONE ", TAG, ": ", nrow(B$X), " x ", ncol(B$X), ", patient files ", length(fs), ", thinned ", sum(D$thinned), ", QC pass ", sum(D$qc_pass),
          ", max |block - stored pkg| ", signif(max(abs(B$X - REF)), 3), ", all-zero columns ", sum(colSums(B$X != 0) == 0))
} else stop("unknown MODE ", MODE_CND)
