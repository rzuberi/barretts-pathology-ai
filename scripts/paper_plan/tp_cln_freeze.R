# Two papers, agent CLN (docs/two_papers_CLN.md): frozen models for bundle v2. kc_freeze.R (their copy-number matrix) or kv_freeze.R (package features) is
# executed unchanged except for the image input: IMG_FEAT = a sample x 1536 mean-embedding table (ART's clean-tile features,
# killcoyne_mm/two_papers_ART/feats/clean.csv.gz). With IMG_FEAT unset the scripts run on their original input (img_mean.csv) and must reproduce the
# committed v1 models byte for byte (smoke). SCRIPT = kc | kv; OUT = output folder.
SCRIPT <- Sys.getenv("SCRIPT"); IMG_PATH <- Sys.getenv("IMG_FEAT", ""); stopifnot(SCRIPT %in% c("kc", "kv"))
src <- readLines(file.path("/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/scripts/paper_plan", paste0(SCRIPT, "_freeze.R")))
swap1 <- function(src, a, b) { i <- grep(a, src, fixed = TRUE); stopifnot(length(i) == 1); src[i] <- sub(a, b, src[i], fixed = TRUE); src }
if (nzchar(IMG_PATH)) {
  stopifnot(file.exists(IMG_PATH))
  if (SCRIPT == "kc") src <- swap1(src, 'file.path(M, "img_mean.csv")', "IMG_PATH") else src <- swap1(src, "IM <- load_img()", "IM <- load_img_cln()")
}
load_img_cln <- function() { R <- readr::read_csv(IMG_PATH, col_types = readr::cols(Sample = "c", .default = "d"), progress = FALSE); X <- as.matrix(R[, -1]); rownames(X) <- R$Sample; stopifnot(all(ids %in% rownames(X))); X[ids, ] }
eval(parse(text = src))
message("TPCLN FREEZE.R DONE ", SCRIPT, " ", if (nzchar(IMG_PATH)) IMG_PATH else "img_mean.csv")
