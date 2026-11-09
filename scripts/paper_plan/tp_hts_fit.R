# Two papers, agent HTS (docs/two_papers_HTS.md): do only some tiles drive the artefact-free H&E score (clean-tile L-IMG)?  Step 2 of 3, CPU only.
# Inputs: tp_hts_prep.py's kept-tile embeddings; the stored outer-fit coefficients of the clean refit (two_papers_ART/refit/clean_coef, tp_art_refit.R
# FEAT=clean_coef COEF=1, inner_fold 0 = the outer fit) and its stored held-out predictions (two_papers_ART/refit/clean).
# Tile score u_t = c0 + w . e_t (raw scale), so the held-out log-odds = mean of u_t over the sample's kept tiles (as op_fig3_tiles.py).
# A (no retraining): held-out samples of fold k scored by the fold-k model on a tile subset: top / bottom k by u, most extreme k by |u - sample median|,
#   random k (20 separate draws, AUROC averaged over draws in tp_hts_eval.py), k = 1, 4, 16, 64 (all kept tiles if fewer); all kept tiles = the stored score. Plus concentration shares (C).
# B (retraining): in outer fold k, stage 1 = the stored fold-k model (trained on the training patients only) gives u_t for every tile of every sample;
#   each sample's mean embedding is taken over its top k (or most extreme k, or random k) tiles; the L-IMG recipe (kv_common.R cvcurves, training-row
#   z-scaling as tp_art_refit.R) is refitted on the training rows and scores the held-out rows. k = 4, 16, 64. "full" = refit on all kept tiles (smoke:
#   must equal the stored prediction).
# REP=1..10 [FOLDS=1,2,..] -> killcoyne_mm/two_papers_HTS/fit/{A,B}_rep_XX[_smoke].csv (row-level, cluster only).
source("/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/scripts/paper_plan/kv_common.R")
REP <- as.integer(Sys.getenv("REP")); FS <- Sys.getenv("FOLDS", ""); FOLDS <- if (nzchar(FS)) as.integer(strsplit(FS, ",")[[1]]) else 1:10
SUF <- if (nzchar(FS)) "_smoke" else ""; W <- file.path(M, "two_papers_HTS"); ART <- file.path(M, "two_papers_ART"); dir.create(file.path(W, "fit"), showWarnings = FALSE)
fa_ <- file.path(W, "fit", sprintf("A_rep_%02d%s.csv", REP, SUF)); fb_ <- file.path(W, "fit", sprintf("B_rep_%02d%s.csv", REP, SUF))
if (file.exists(fa_) && file.exists(fb_)) quit(save = "no")
DOB <- !file.exists(fb_)   # B is skipped when its file exists (A rerun alone)
# --- copied verbatim from tp_art_refit.R
scale_tr <- function(Mx, ref) {
  mi <- colMeans(Mx[ref, , drop = FALSE], na.rm = TRUE); if (anyNA(Mx)) { w <- which(is.na(Mx), arr.ind = TRUE); Mx[w] <- mi[w[, 2]] }
  mu <- colMeans(Mx[ref, , drop = FALSE]); sdv <- apply(Mx[ref, , drop = FALSE], 2, sd); Z <- sweep(sweep(Mx, 2, mu), 2, ifelse(sdv > 0, sdv, 1), "/"); Z[, sdv == 0] <- 0
  list(Z = Z, mu = mu, sdv = sdv, mi = mi) }
strat_folds <- function(pp, yy, seed, nf) { pl <- tapply(yy, pp, max); Pp <- names(pl)[pl == 1]; Np <- names(pl)[pl == 0]; set.seed(seed); ord <- c(sample(Pp), sample(Np)); setNames(rep(1:nf, length.out = length(ord)), ord) }
# --- inputs
IX <- read_csv(file.path(W, "kept_index.csv"), col_types = cols(sample = "c", .default = "d")); NT <- sum(IX$n_kept)
fe_ <- file.path(W, "emb_kept.f32"); stopifnot(file.size(fe_) == NT * 1536 * 4); E <- matrix(readBin(fe_, "numeric", n = NT * 1536, size = 4), nrow = NT)
mi_ <- match(ids, IX$sample); stopifnot(!anyNA(mi_)); TI <- lapply(mi_, function(j) IX$start[j] + seq_len(IX$n_kept[j]))
CO <- read_csv(file.path(ART, "refit", "clean_coef", sprintf("coef_rep_%02d.csv.gz", REP)), show_col_types = FALSE); CO <- CO[CO$inner_fold == 0, ]; stopifnot(nrow(CO) == 10)
wc <- paste0("w", 1:1536); SP <- read_csv(file.path(ART, "refit", "clean", sprintf("outer_rep_%02d.csv", REP)), col_types = cols(Sample = "c", .default = "d"))
SP <- SP[match(ids, SP$Sample), ]; fold_of <- strat_folds(pat, y, REP, 10); stopifnot(all(SP$fold == fold_of[pat]))
CLN <- t(vapply(TI, function(r) colMeans(E[r, , drop = FALSE]), numeric(1536)))
pick <- function(rule, u, kk) { nk <- min(kk, length(u)); switch(rule, top = order(-u)[1:nk], bot = order(u)[1:nk], ext = order(-abs(u - median(u)))[1:nk],
                                                                  rnd = sample.int(length(u), nk)) }
share10 <- function(a) { a <- sort(a, decreasing = TRUE); sum(a[seq_len(ceiling(0.1 * length(a)))]) / sum(a) }
KA <- c(1, 4, 16, 64); KB <- c(4, 16, 64); RB <- c("top", "ext", "rnd"); outA <- list(); outB <- list()
for (k in FOLDS) {
  t0 <- Sys.time(); te <- which(fold_of[pat] == k); tr <- which(fold_of[pat] != k)
  c0 <- CO$c0[CO$fold == k]; w <- as.numeric(unlist(CO[CO$fold == k, wc])); U <- c0 + as.vector(E %*% w)
  # A: same fitted model, tile subsets of the held-out samples
  set.seed(100000 * REP + k); ra <- list()
  for (i in te) {
    u <- U[TI[[i]]]; r <- list(Sample = ids[i], rep = REP, fold = k, n_kept = length(u), all = mean(u),
                              stored_logit = qlogis(SP$pred[i]), share_c0 = share10(abs(u - c0)), share_med = share10(abs(u - median(u))))
    for (kk in KA) { for (rl in c("top", "bot", "ext")) r[[paste0(rl, "_", kk)]] <- mean(u[pick(rl, u, kk)])
      for (d in 1:20) r[[sprintf("rnd_%d_d%02d", kk, d)]] <- mean(u[pick("rnd", u, kk)]) }   # 20 single random draws; AUROC averaged over draws
    ra[[length(ra) + 1]] <- as.data.frame(r) }
  outA[[length(outA) + 1]] <- bind_rows(ra)
  # B: select with the stage-1 (fold-k) model for every sample, refit on the training rows only, score the held-out rows
  if (!DOB) { message(sprintf("HTS rep %d fold %d: A all-vs-stored max %.2e", REP, k, max(abs(outA[[length(outA)]]$all - outA[[length(outA)]]$stored_logit)))); next }
  rb <- data.frame(Sample = ids[te], rep = REP, fold = k, stored = SP$pred[te])
  fitp <- function(X) { sc <- scale_tr(X, tr); cv <- cvcurves(sc$Z[tr, ], y[tr], pat[tr]); predict(cv$fit, sc$Z[te, , drop = FALSE], s = cv$class_min, type = "response")[, 1] }
  rb$full <- fitp(CLN)
  for (rl in RB) for (kk in KB) {
    set.seed(200000 * REP + 100 * k + kk)
    X <- t(vapply(TI, function(r) colMeans(E[r[pick(rl, U[r], kk)], , drop = FALSE]), numeric(1536))); rb[[paste0(rl, "_", kk)]] <- fitp(X) }
  outB[[length(outB) + 1]] <- rb
  message(sprintf("HTS rep %d fold %d: A all-vs-stored max %.2e, B full-vs-stored max %.2e, %.0f s", REP, k,
                  max(abs(outA[[length(outA)]]$all - outA[[length(outA)]]$stored_logit)), max(abs(rb$full - rb$stored)), as.numeric(Sys.time() - t0, units = "secs")))
}
write_csv(bind_rows(outA), fa_); if (DOB) write_csv(bind_rows(outB), fb_); message("HTS FIT DONE rep ", REP)
