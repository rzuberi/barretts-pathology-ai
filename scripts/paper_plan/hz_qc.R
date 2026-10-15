# Horizons H2 (docs/paper_survival_horizons.md @ ee51db8): export the package QC table (segmentRawData residuals) per published sample from
# feasibility/paper_plan/killcoyne/pkg/pat_*.rds (kr_package.R). Row-level output stays on the cluster: killcoyne_mm/horizons/pkg_qc.csv.
suppressMessages({library(readr); library(dplyr)})
K <- "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/feasibility/paper_plan/killcoyne"
out <- list()
for (f in sort(Sys.glob(file.path(K, "pkg", "pat_*.rds")))) {
  o <- readRDS(f); if (isTRUE(o$error)) next
  q <- as.data.frame(o$qc); id <- intersect(c("sample", "Sample", "samplename"), colnames(q))[1]
  q$sid <- as.character(q[[id]]); q <- left_join(q, as.data.frame(o$map), by = "sid"); out[[f]] <- q
}
Q <- bind_rows(out); write_csv(Q, file.path(K, "../killcoyne_mm/horizons/pkg_qc.csv")); message("HZ QC DONE rows ", nrow(Q), " cols ", paste(colnames(Q), collapse = ","))
