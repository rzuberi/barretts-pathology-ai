suppressMessages({library(QDNAseq); library(Biobase)})
D <- "/mnt/scratche/fast/fmlab/datasets/imaging/SWGCohort/copy_number_hg38/train/perPatient/50kb/SLX-12453.D701_D507"
rc <- readRDS(file.path(D, "50.read_counts.rds")); print(class(rc)); print(assayDataElementNames(rc)); print(head(fData(rc))); print(table(fData(rc)$use)); print(pData(rc))
e <- new.env(); load(file.path(D, "50.QDNAseq.RData"), envir = e); print(ls(e)); for (v in ls(e)) { cat(v, class(get(v, e)), "\n") }
for (v in ls(e)) { o <- get(v, e); if (is(o, "QDNAseqSignals") || is(o, "QDNAseqReadCounts")) { cat("--", v, "\n"); print(assayDataElementNames(o)); print(table(fData(o)$use)); print(pData(o)) } }
for (v in ls(e)) { o <- get(v, e); if (is.character(o) || is.numeric(o) || is.logical(o)) if (length(o) < 20) { cat(v, ": "); print(o) } }
fit <- read.delim(file.path(D, "50.fitted_read_counts.txt")); raw <- read.delim(file.path(D, "50.raw_read_counts.txt"))
cat("fit NA", sum(is.na(fit[,5])), "raw sum", sum(raw[,5]), "rc sum", sum(assayDataElement(rc, "counts")), "\n")
# try: estimateCorrection on stored read counts reproduces stored fit?
rcf <- applyFilters(rc, residual = TRUE, blacklist = TRUE); ec <- estimateCorrection(rcf); f2 <- assayDataElement(ec, "fit")[, 1]
cat("default filters: max abs diff vs stored fit", max(abs(f2 - fit[,5]), na.rm = TRUE), "NA match", identical(is.na(f2), is.na(fit[,5])), "\n")
ec0 <- estimateCorrection(rc); f0 <- assayDataElement(ec0, "fit")[, 1]
cat("stored use: max abs diff", max(abs(f0 - fit[,5]), na.rm = TRUE), "NA match", identical(is.na(f0), is.na(fit[,5])), "\n")
