# Demographics completion D2 (docs/demographics_completion.md @ 24b2972): data objects shipped in BarrettsProgressionRisk (data/ and sysdata), file listing
# of the two local clones. Output: feasibility/paper_plan/demographics/probe3_pkg.txt
suppressMessages(library(BarrettsProgressionRisk)); O <- "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/feasibility/paper_plan/demographics/probe3_pkg.txt"; sink(O)
d <- data(package = "BarrettsProgressionRisk")$results; cat("== data():\n"); print(d[, c("Item", "Title")])
e <- new.env(); for (it in d[, "Item"]) { data(list = it, package = "BarrettsProgressionRisk", envir = e) }
for (nm in ls(e)) { x <- get(nm, e); cat("\n--", nm, class(x)[1], if (!is.null(dim(x))) paste(dim(x), collapse = "x") else length(x), "\n"); if (is.data.frame(x) || is.matrix(x)) print(head(colnames(x), 60)) else if (is.list(x)) print(names(x)) }
ns <- asNamespace("BarrettsProgressionRisk"); cat("\n== namespace objects that are data:\n")
for (nm in ls(ns, all.names = TRUE)) { x <- get(nm, ns); if (is.data.frame(x) || is.matrix(x) || (is.list(x) && !is.function(x))) { cat("--", nm, class(x)[1], if (!is.null(dim(x))) paste(dim(x), collapse = "x") else length(x), "\n"); if (is.data.frame(x) || is.matrix(x)) print(head(colnames(x), 60)) else print(head(names(x), 40)) } }
for (cl in c("/mnt/scratche/slow/fmlab/zuberi01/phd/BarrettsProgressionRisk", "/mnt/scratche/slow/fmlab/zuberi01/phd/barretts_retraining/paper/qc_historical_replay_plan_20260330/BarrettsProgressionRisk_repo")) {
  cat("\n== clone", cl, "\n"); f <- list.files(cl, recursive = TRUE, all.files = FALSE); f <- f[!grepl("^\\.git/", f)]; print(f) }
sink(); message("PROBE3 DONE")
