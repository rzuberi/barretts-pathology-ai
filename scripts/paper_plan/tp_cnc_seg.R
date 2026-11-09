# Two papers, agent CNC (docs/two_papers_CNC.md): per-sample copy-number processing of the set C samples from the stored per-sample QDNAseq outputs
# (SWGCohort/copy_number_hg38/train/perPatient/50kb/<sample>/, read-only; kr_samples.csv raw/fit paths). For each sample:
#   - check: raw counts in 50.read_counts.rds equal 50.raw_read_counts.txt (the file ACQ matched bin by bin to the BAMs);
#   - CBS: the stored QDNAseq CBS segmentation (50.copy_number_segmented.rds), ACE template (bins in use), segments as ratio to the sample median;
#   - ACE: singlemodel (ploidy 2, RMSE, penalty 0, cellularity 5-100%), best fit = lowest relative error; getadjustedsegments -> absolute copies;
#   - ichorCNA: raw 50 kb counts summed into the 1 Mb bins of ichorCNA's hg38 GC / mappability wigs, runIchorCNA.R (broadinstitute master), no panel
#     of normals, autosomes, ploidy c(2,3), normal c(0.5 ... 0.99), maxCN 5, no subclonal states, low-tumour-fraction transition settings.
# Autosomes only (as the package pipeline). IDX = 1..676 (set C samples in kr_samples.csv order) or SMOKE=1 (the first 5).
# -> killcoyne_mm/two_papers_CNC/seg/s_XXX.rds (or smoke.rds): list(sample -> list(check, cbs, ace, ace_fit, ichor, ichor_params)). Row-level, cluster only.
.libPaths(c("/mnt/scratche/slow/fmlab/zuberi01/envs/cnc_rlib", .libPaths()))
suppressMessages({library(QDNAseq); library(Biobase); library(ACE)})
K <- "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/feasibility/paper_plan/killcoyne"; M <- paste0(K, "_mm"); OD <- file.path(M, "two_papers_CNC", "seg"); dir.create(OD, recursive = TRUE, showWarnings = FALSE)
ICH <- "/mnt/scratche/slow/fmlab/zuberi01/envs/cnc_tools/ichor/runIchorCNA.R"; EXT <- system.file("extdata", package = "ichorCNA")
RS <- "/mnt/scratche/slow/fmlab/zuberi01/envs/killcoyne_r/bin/Rscript"
smp <- read.csv(file.path(K, "kr_samples.csv"), colClasses = "character"); C <- read.csv(file.path(M, "set_C.csv"), colClasses = "character")
smp <- smp[smp$Sample %in% C$Sample, ]; stopifnot(nrow(smp) == nrow(C))
SMOKE <- Sys.getenv("SMOKE", "0") == "1"; IDX <- as.integer(Sys.getenv("IDX", "-1"))   # IDX = row of the set C samples in kr_samples.csv order (1..676)
s <- if (SMOKE) smp[1:5, ] else smp[IDX, ]; stopifnot(nrow(s) > 0)
outf <- file.path(OD, if (SMOKE) "smoke.rds" else sprintf("s_%03d.rds", IDX)); if (file.exists(outf)) quit(save = "no")
segs_from <- function(chr, st, en, v) {   # run-length segments of a per-bin segment value (consecutive equal values within a chromosome)
  key <- paste(chr, v); r <- rle(key); e <- cumsum(r$lengths); b <- e - r$lengths + 1
  data.frame(chr = chr[b], start = st[b], end = en[e], n_bins = r$lengths, value = v[b], stringsAsFactors = FALSE) }
gcw <- file.path(EXT, "gc_hg38_1000kb.wig"); mpw <- file.path(EXT, "map_hg38_1000kb.wig"); cen <- file.path(EXT, "GRCh38.GCA_000001405.2_centromere_acen.txt")
stopifnot(file.exists(gcw), file.exists(mpw), file.exists(cen))
ucsc <- grepl("chrom=chr", readLines(gcw, n = 1))
write_wig <- function(raw, f) {   # 50 kb counts -> 1 Mb fixedStep wig aligned to ichorCNA's bins (start 1, step 1e6)
  raw <- raw[raw$chromosome %in% as.character(1:22), ]; raw$b <- (raw$start - 1) %/% 1e6
  con <- file(f, "w")
  for (ch in as.character(1:22)) {
    r <- raw[raw$chromosome == ch, ]; nb <- max(r$b) + 1; v <- tapply(r$count, factor(r$b, levels = 0:(nb - 1)), sum); v[is.na(v)] <- 0
    writeLines(sprintf("fixedStep chrom=%s%s start=1 step=1000000 span=1000000", if (ucsc) "chr" else "", ch), con); writeLines(as.character(as.integer(v)), con) }
  close(con) }
out <- list()
for (j in seq_len(nrow(s))) {
  d <- dirname(s$raw[j]); sm <- s$Sample[j]; t0 <- Sys.time(); o <- list(pat_index = s$pat_index[j])
  # --- check: stored raw counts
  rc <- readRDS(file.path(d, "50.read_counts.rds")); rt <- read.delim(s$raw[j], check.names = FALSE, colClasses = c(chrom = "character"))
  o$check <- list(n_bins = nrow(rt), counts_equal = isTRUE(all(as.numeric(assayDataElement(rc, "counts")[, 1]) == as.numeric(rt[[5]]))), total = sum(as.numeric(rt[[5]])))
  # --- CBS (stored) and ACE
  cn <- readRDS(file.path(d, "50.copy_number_segmented.rds")); tp <- objectsampletotemplate(cn, 1); tn <- na.exclude(tp); tn <- tn[tn$chr %in% as.character(1:22), ]
  med <- median(tn$segments); cb <- segs_from(as.character(tn$chr), tn$start, tn$end, tn$segments); cb$value <- cb$value / med; o$cbs <- cb   # ratio to the median (log2 taken downstream, after tiling)
  o$cbs_bins <- data.frame(chr = as.character(tn$chr), start = tn$start, end = tn$end, cn = tn$copynumbers / med, seg = tn$segments / med)
  fit <- tryCatch(singlemodel(tp, ploidy = 2, method = "RMSE", penalty = 0, highlightminima = FALSE), error = function(e) NULL)
  if (!is.null(fit)) {
    bi <- which.min(fit$rerror); cell <- fit$minima[bi]; o$ace_fit <- list(cellularity = cell, rel_error = fit$rerror[bi], n_minima = length(fit$minima), minima = fit$minima, rerror = fit$rerror)
    a <- getadjustedsegments(tp, cellularity = cell, ploidy = 2); a <- a[a$Chromosome %in% as.character(1:22), ]
    o$ace <- data.frame(chr = as.character(a$Chromosome), start = a$Start, end = a$End, n_bins = a$Num_Bins, value = a$Segment_Mean2, copies = a$Copies)
  }
  # --- ichorCNA
  td <- file.path(tempdir(), sprintf("ich_%d", j)); dir.create(td); wf <- file.path(td, "t.wig")
  write_wig(data.frame(chromosome = as.character(rt[[2]]), start = as.numeric(rt[[3]]), count = as.numeric(rt[[5]])), wf)
  args <- c(ICH, "--id", "s", "--WIG", wf, "--gcWig", gcw, "--mapWig", mpw, "--centromere", cen, "--genomeBuild", "hg38", "--genomeStyle", if (ucsc) "UCSC" else "NCBI",
            "--chrs", "'c(1:22)'", "--chrTrain", "'c(1:22)'", "--chrNormalize", "'c(1:22)'", "--ploidy", "'c(2,3)'", "--normal", "'c(0.5,0.6,0.7,0.8,0.9,0.95,0.99)'",
            "--maxCN", "5", "--estimateScPrevalence", "FALSE", "--scStates", "'c()'", "--includeHOMD", "FALSE", "--txnE", "0.9999", "--txnStrength", "10000", "--outDir", td)
  lg <- suppressWarnings(system2(RS, args, stdout = TRUE, stderr = TRUE, env = "R_LIBS=/mnt/scratche/slow/fmlab/zuberi01/envs/cnc_rlib")); sf <- file.path(td, "s.seg"); pf <- file.path(td, "s.params.txt")
  if (file.exists(sf)) {
    g <- read.delim(sf, check.names = FALSE, stringsAsFactors = FALSE); g <- g[sub("^chr", "", g$chr) %in% as.character(1:22), ]
    o$ichor <- data.frame(chr = sub("^chr", "", g$chr), start = g$start, end = g$end, n_bins = g$bins, value = g$median, copies = g$copy.number)
    pl <- readLines(pf); grab <- function(k) as.numeric(sub(".*:\\s*", "", grep(k, pl, value = TRUE)[1]))
    o$ichor_params <- list(tumor_fraction = grab("^Tumor Fraction"), ploidy = grab("^Ploidy"))
  } else o$ichor_error <- tail(lg, 5)
  unlink(td, recursive = TRUE); o$secs <- as.numeric(difftime(Sys.time(), t0, units = "secs")); out[[sm]] <- o
  message(sm, " cbs ", nrow(o$cbs), " ace ", if (is.null(o$ace)) NA else nrow(o$ace), " cell ", if (is.null(o$ace_fit)) NA else o$ace_fit$cellularity,
          " ichor ", if (is.null(o$ichor)) NA else nrow(o$ichor), " tf ", if (is.null(o$ichor_params)) NA else o$ichor_params$tumor_fraction, " secs ", round(o$secs))
}
saveRDS(out, outf); message("TP CNC SEG DONE ", basename(outf))
