# Two papers, agent CNC (docs/two_papers_CNC.md): probe the stored per-sample QDNAseq outputs (object classes, slots, options); prints structure only.
suppressMessages({library(QDNAseq); library(Biobase)})
K <- "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/feasibility/paper_plan/killcoyne"
smp <- read.csv(file.path(K, "kr_samples.csv"), colClasses = "character"); d <- dirname(smp$raw[1])
cat("n sheet rows", nrow(smp), "\n")
for (f in c("50.copy_number_segmented.rds", "50.read_counts.rds")) { o <- readRDS(file.path(d, f)); cat(f, class(o), "\n"); print(assayDataElementNames(o)); print(dim(o)); print(head(fData(o), 3)); print(pData(o)) }
e <- new.env(); load(file.path(d, "50.QDNAseq.RData"), envir = e); cat("RData objects:", ls(e), "\n"); for (v in ls(e)) cat(v, class(e[[v]]), "\n")
