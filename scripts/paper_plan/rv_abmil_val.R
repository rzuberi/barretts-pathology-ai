# R13 of docs/paper_review_round1_analyses.md (baa47ab): validation (early-stopping) patients for N-IMG in every repeat x fold of the kv_cv.R
# folds, chosen as in km_neural.py: group 1 of the r = 1 shuffle of the training patients (km_linear.R shuffle_groups, unchanged), patient order
# = set C order. Check first: the same function reproduces the stored LOPO neural_val_patients.csv exactly. Base R only.
# Out (cluster only): feasibility/paper_plan/killcoyne_mm/review_round1/abmil/val_patients.csv (rep, fold, val_patient).
M <- "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/feasibility/paper_plan/killcoyne_mm"; O <- file.path(M, "review_round1", "abmil")
dir.create(O, recursive = TRUE, showWarnings = FALSE)
shuffle_groups <- function(pp, r) { up <- unique(pp); set.seed(r); sp <- sample(up); setNames(rep(1:5, length.out = length(sp)), sp) }
C <- read.csv(file.path(M, "set_C.csv"), colClasses = "character"); pat <- C$Patient
# check: LOPO validation patients of km_neural.py reproduced
old <- read.csv(file.path(M, "neural_val_patients.csv"), colClasses = "character")
chk <- do.call(rbind, lapply(sort(unique(pat)), function(p) { g <- shuffle_groups(pat[pat != p], 1); data.frame(left_out = p, val_patient = names(g)[g == 1]) }))
stopifnot(nrow(chk) == nrow(old), all(chk$left_out == old$left_out), all(chk$val_patient == old$val_patient)); message("LOPO validation patients reproduced: ", nrow(old), " rows")
out <- list()
for (r in 1:10) {
  F <- read.csv(file.path(M, "cv", "preds", sprintf("cfg_00_rep_%02d.csv", r)), colClasses = "character")
  fo <- tapply(F$fold, F$Patient, function(v) { stopifnot(length(unique(v)) == 1); v[1] }); stopifnot(all(pat %in% names(fo)))
  for (k in 1:10) {
    trp <- pat[fo[pat] != as.character(k)]; g <- shuffle_groups(trp, 1)
    out[[length(out) + 1]] <- data.frame(rep = r, fold = k, val_patient = names(g)[g == 1])
  }
}
V <- do.call(rbind, out); write.csv(V, file.path(O, "val_patients.csv"), row.names = FALSE, quote = FALSE)
message("VAL DONE ", nrow(V), " rows; per fold ", paste(range(table(paste(V$rep, V$fold))), collapse = "-"))
