# Killcoyne protocol with imaging, linear tier (docs/paper_plan_killcoyne_multimodal.md @ 3904c44). glmnet alpha 0.9, standardize = FALSE,
# LOPO over the 80 patients of set C; per left-out patient lambda by 10 x 5 patient-grouped CV (set.seed(r), r = 1..10); rules class_min
# (primary), dev_min, global. CFG_ID picks model x CNV source; CHUNK_ID / N_CHUNKS shard patients. EXPORT=1 writes the neural-tier inputs
# (package CNV block for C, and each left-out patient's validation patients = group 1 of the r = 1 shuffle) and quits.
suppressMessages({library(glmnet); library(readr); library(dplyr)})
K <- "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/feasibility/paper_plan/killcoyne"; M <- paste0(K, "_mm")
grid <- data.frame(model = c("cnv", "img", "early", "inter", "cnv", "early", "inter"), cnvsrc = c("their", "none", "their", "their", "pkg", "pkg", "pkg"), stringsAsFactors = FALSE)
C <- read_csv(file.path(M, "set_C.csv"), col_types = cols(.default = "c")); ids <- C$Sample; y <- as.integer(C$Status == "P"); pat <- C$Patient; n <- length(ids)
zs <- function(Mx, ref) { mu <- colMeans(Mx[ref, , drop = FALSE]); sdv <- apply(Mx[ref, , drop = FALSE], 2, sd); Z <- sweep(sweep(Mx, 2, mu), 2, ifelse(sdv > 0, sdv, 1), "/"); Z[, sdv == 0] <- 0; Z }
cnv_block <- function(src) {
  if (src == "their") { R <- read_csv(file.path(K, "their_features.csv"), col_types = cols(Sample = "c", .default = "d")); X <- as.matrix(R[, -1]); rownames(X) <- R$Sample; return(X[ids, ]) }
  P <- readRDS(file.path(K, "pkg_tiles.rds")); loc <- function(nm) { a <- do.call(rbind, strsplit(sub("-", ":", nm), ":")); data.frame(chr = a[, 1], s = as.numeric(a[, 2]), e = as.numeric(a[, 3])) }
  lt <- loc(colnames(P$seg)); la <- loc(colnames(P$arm)); W <- matrix(0, ncol(P$arm), ncol(P$seg)); for (i in seq_len(nrow(la))) W[i, lt$chr == la$chr[i] & lt$s <= la$e[i] & lt$e >= la$s[i]] <- 1
  ref <- seq_len(n); S <- zs(P$seg[ids, ], ref); A <- zs(P$arm[ids, ], ref); cx <- apply(S, 1, function(x) sum(x >= mean(x) + 2 * sd(x) | x <= mean(x) - 2 * sd(x))); cx <- (cx - mean(cx)) / sd(cx)
  X <- cbind(S - A %*% W, A, cx = cx); X[!is.finite(X)] <- 0; X }   # reconciliation plan B construction, standardised over C
shuffle_groups <- function(pp, r) { up <- unique(pp); set.seed(r); sp <- sample(up); setNames(rep(1:5, length.out = length(sp)), sp) }
if (Sys.getenv("EXPORT") == "1") {
  Xp <- cnv_block("pkg"); write_csv(as_tibble(Xp) %>% mutate(sample_id = ids, .before = 1), file.path(M, "cnv_pkg_C.csv"))
  Xt <- cnv_block("their"); write_csv(as_tibble(Xt) %>% mutate(sample_id = ids, .before = 1), file.path(M, "cnv_their_C.csv"))
  up <- sort(unique(pat)); v <- bind_rows(lapply(up, function(p) { g <- shuffle_groups(pat[pat != p], 1); tibble(left_out = p, val_patient = names(g)[g == 1]) }))
  write_csv(v, file.path(M, "neural_val_patients.csv")); message("EXPORT DONE"); quit(save = "no")
}
cfg <- as.integer(Sys.getenv("CFG_ID", "0")); ch <- as.integer(Sys.getenv("CHUNK_ID", "0")); nch <- as.integer(Sys.getenv("N_CHUNKS", "1")); g <- grid[cfg + 1, ]
dir.create(file.path(M, "linear"), showWarnings = FALSE); outf <- file.path(M, "linear", sprintf("cfg_%02d_chunk_%02d.csv", cfg, ch)); if (file.exists(outf)) quit(save = "no")
message("cfg ", cfg, " model ", g$model, " cnv ", g$cnvsrc, " rows ", n, " patients ", length(unique(pat)))
IM <- NULL; if (g$model != "cnv") { R <- read_csv(file.path(M, "img_mean.csv"), col_types = cols(Sample = "c", .default = "d")); IM <- as.matrix(R[, -1]); rownames(IM) <- R$Sample; IM <- zs(IM[ids, ], seq_len(n)) }
CN <- if (g$model != "img") cnv_block(g$cnvsrc) else NULL
pca_block <- function(B, tr, k = 32) { pc <- prcomp(B[tr, ], center = TRUE, scale. = FALSE, rank. = k); Z <- predict(pc, B); zs(Z, tr) }
features <- function(tr) switch(g$model, cnv = CN, img = IM, early = cbind(CN, IM), inter = cbind(pca_block(CN, tr), pca_block(IM, tr)))
cvcurves <- function(X, yy, pp) {
  fit <- glmnet(X, yy, family = "binomial", alpha = 0.9, standardize = FALSE); lam <- fit$lambda; cls <- dev <- matrix(NA, 10, length(lam))
  for (r in 1:10) {
    fo <- shuffle_groups(pp, r)[pp]; fe <- fd <- matrix(NA, 5, length(lam)); w <- numeric(5)
    for (k in 1:5) { te <- fo == k; f <- glmnet(X[!te, ], yy[!te], family = "binomial", alpha = 0.9, lambda = lam, standardize = FALSE); pr <- pmin(pmax(predict(f, X[te, , drop = FALSE], type = "response"), 1e-5), 1 - 1e-5)
      if (ncol(pr) < length(lam)) pr <- cbind(pr, matrix(pr[, ncol(pr)], nrow(pr), length(lam) - ncol(pr)))
      fe[k, ] <- colMeans((pr > 0.5) != yy[te]); fd[k, ] <- colMeans(-2 * (yy[te] * log(pr) + (1 - yy[te]) * log(1 - pr))); w[k] <- sum(te) }
    cls[r, ] <- colSums(fe * w) / sum(w); dev[r, ] <- colSums(fd * w) / sum(w) }
  cm <- colMeans(cls); dm <- colMeans(dev); list(fit = fit, class_min = max(lam[cm <= min(cm)]), dev_min = lam[which.min(dm)]) }
gl <- cvcurves(features(seq_len(n)), y, pat)$class_min
up <- sort(unique(pat)); mine <- up[(seq_along(up) - 1) %% nch == ch]; out <- list()
for (p in mine) {
  te <- pat == p; tr <- which(!te); X <- features(tr); cv <- cvcurves(X[tr, ], y[tr], pat[tr])
  lams <- c(class_min = cv$class_min, dev_min = cv$dev_min, global = gl); pr <- predict(cv$fit, X[te, , drop = FALSE], s = lams, type = "response")
  for (j in seq_along(lams)) out[[length(out) + 1]] <- data.frame(Sample = ids[te], Patient = p, y = y[te], cfg = cfg, model = g$model, cnvsrc = g$cnvsrc, rule = names(lams)[j], lambda = lams[j], pred = pr[, j])
}
write_csv(bind_rows(out), outf); message("LINEAR CHUNK DONE cfg ", cfg, " chunk ", ch)
