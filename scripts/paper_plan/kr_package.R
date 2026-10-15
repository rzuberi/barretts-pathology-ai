# Killcoyne reconciliation plan A, step 1 (docs/paper_plan_killcoyne_reconcile.md @ a655e08): run the BarrettsProgressionRisk
# segmentation on our QDNAseq 50 kb raw + fitted counts, one patient at a time (multipcf across that patient's samples, as the package
# does), package defaults except build='hg38' and the blacklist lifted to hg38 (kr_prep.py). Saves unscaled 5-Mb and arm tiles for all
# segmented samples (QC pass and fail, with the QC table) per patient. Sharded by SHARD_ID / N_SHARDS over patient index.
suppressMessages({library(BarrettsProgressionRisk); library(tidyverse)})
# package bug: chrInfo() reads the bundled <build>_info.txt only when verbose=TRUE, otherwise it downloads UCSC files whose current format
# no longer parses; segmentRawData/tileSegments call it with verbose=FALSE. Replace it with a reader of the bundled file.
chrInfo_bundled <- function(chrs = c(1:22, "X", "Y"), prefix = "chr", build = "hg19", file = NULL, verbose = FALSE) {
  f <- system.file("extdata", paste0(build, "_info.txt"), package = "BarrettsProgressionRisk")
  read.table(f, header = TRUE, sep = "\t", stringsAsFactors = FALSE) %>% as_tibble() %>% mutate(chr = sub(prefix, "", chrom)) %>% mutate(chr = factor(chr, levels = chrs, ordered = TRUE))
}
assignInNamespace("chrInfo", chrInfo_bundled, ns = "BarrettsProgressionRisk")
# copynumber::pcf/multipcf accept only hg16-hg19 as 'assembly' (used only to split at centromeres); for hg38 pass the arms explicitly,
# computed from the package's bundled hg38 centromeres (p if position < centromere midpoint, else q).
h38 <- chrInfo_bundled(build = "hg38")
arms38 <- function(d) { cen <- setNames(h38$chr.cent, h38$chr)[as.character(d[[1]])]; ifelse(d[[2]] < cen, "p", "q") }
orig_mpcf <- copynumber::multipcf; orig_pcf <- copynumber::pcf
mpcf38 <- function(data, ..., assembly = "hg19") { if (assembly == "hg38") orig_mpcf(data, arms = arms38(data), ..., assembly = "hg19") else orig_mpcf(data, ..., assembly = assembly) }
pcf38 <- function(data, ..., assembly = "hg19") { if (assembly == "hg38") orig_pcf(data, arms = arms38(data), ..., assembly = "hg19") else orig_pcf(data, ..., assembly = assembly) }
assignInNamespace("multipcf", mpcf38, ns = "copynumber"); assignInNamespace("pcf", pcf38, ns = "copynumber")
O <- "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/feasibility/paper_plan/killcoyne"
sh <- as.integer(Sys.getenv("SHARD_ID", "0")); ns <- as.integer(Sys.getenv("N_SHARDS", "1"))
smp <- read_csv(file.path(O, "kr_samples.csv"), col_types = cols(.default = "c")) %>% mutate(gidx = row_number())
bl <- read_tsv(file.path(O, "blacklist_hg38.txt"), col_types = "cii")
dir.create(file.path(O, "pkg"), showWarnings = FALSE)
pats <- sort(unique(as.integer(smp$pat_index))); pats <- pats[pats %% ns == sh]
for (pi in pats) {
  outf <- file.path(O, "pkg", sprintf("pat_%03d.rds", pi)); if (file.exists(outf)) next
  s <- smp %>% filter(as.integer(pat_index) == pi); s$sid <- sprintf("K%04dQ", s$gidx)   # regex-safe ids (package matches samples by regex)
  rd <- NULL; fd <- NULL
  for (j in seq_len(nrow(s))) {
    r <- read_tsv(s$raw[j], col_types = cols(chrom = col_character()), progress = FALSE); f <- read_tsv(s$fit[j], col_types = cols(chrom = col_character()), progress = FALSE)
    colnames(r)[5] <- s$sid[j]; colnames(f)[5] <- s$sid[j]
    if (is.null(rd)) { rd <- r[, 1:5]; fd <- f[, 1:5] } else { rd <- left_join(rd, r[, c(1, 5)], by = "location"); fd <- left_join(fd, f[, c(1, 5)], by = "location") }
  }
  keep <- rd$chrom %in% as.character(1:22); rd <- rd[keep, ]; fd <- fd[keep, ]   # autosomes only: the model has no sex-chromosome features, and the lone X bin that survives QDNAseq filtering breaks multipcf (1-bin arm)
  info <- loadSampleInformation(tibble(Sample = s$sid, Endoscopy = as.integer(s$Endoscopy), Pathology = s$Pathology, `P53 IHC` = as.integer(s$`P53 IHC`)))
  t0 <- Sys.time()
  obj <- tryCatch(suppressWarnings(segmentRawData(info, rd, fd, blacklist = bl, build = "hg38", cache.dir = tempdir(), verbose = FALSE)), error = function(e) { message("SEGMENT FAIL ", pi, ": ", conditionMessage(e)); NULL })
  if (is.null(obj)) { saveRDS(list(pat_index = pi, error = TRUE, map = s[, c("sid", "Sample", "Patient")]), outf); next }
  allseg <- obj$seg.vals; if (!is.null(obj$failedQC)) allseg <- cbind(allseg, obj$failedQC[, setdiff(colnames(obj$failedQC), colnames(allseg)), drop = FALSE])
  obj2 <- obj; obj2$seg.vals <- allseg; obj2$residuals <- obj$residuals %>% mutate(Pass = TRUE)   # tile every segmented sample; QC kept separately
  segt <- suppressWarnings(tileSegments(obj2, size = 5e6, verbose = FALSE)); armt <- suppressWarnings(tileSegments(obj2, size = "arms", verbose = FALSE))
  saveRDS(list(pat_index = pi, error = FALSE, map = s[, c("sid", "Sample", "Patient")], qc = obj$residuals, seg = segt$tiles, arm = armt$tiles,
               seg_err = segt$error, arm_err = armt$error, secs = as.numeric(difftime(Sys.time(), t0, units = "secs"))), outf)
  message("patient ", pi, " samples ", nrow(s), " segmented ", nrow(segt$tiles), " pass ", sum(obj$residuals$Pass), " secs ", round(as.numeric(difftime(Sys.time(), t0, units = "secs"))))
}
message("PKG SHARD DONE ", sh)
