suppressMessages({library(QDNAseq); library(Biobase)})
D <- "/mnt/scratche/fast/fmlab/datasets/imaging/SWGCohort/copy_number_hg38/train/perPatient/50kb/SLX-12453.D701_D507"
rc <- readRDS(file.path(D, "50.read_counts.rds")); fit <- read.delim(file.path(D, "50.fitted_read_counts.txt"))
e <- new.env(); load(file.path(D, "50.QDNAseq.RData"), envir = e); fs <- assayDataElement(e$read_counts_corrected, "fit")[, 1]
cat("stored obj fit vs txt:", max(abs(fs - fit[,5]), na.rm=TRUE), identical(is.na(fs), is.na(fit[,5])), "\n"); print(table(objNA = is.na(fs), txtNA = is.na(fit[,5])))
f2 <- assayDataElement(estimateCorrection(applyFilters(rc, residual = TRUE, blacklist = TRUE)), "fit")[, 1]
print(table(oursNA = is.na(f2), txtNA = is.na(fit[,5]))); cat("max abs diff obj fit vs ours", max(abs(f2 - fs), na.rm = TRUE), "\n")
print(table(use = fData(e$read_counts_corrected)$use, txtNA = is.na(fit[,5])))
