# Two papers, agent CND (docs/two_papers_CND.md): L-CNV recipe refitted on reduced or thinned copy-number blocks under the stored outer folds.
# kv_common.R sourced unchanged (set C, cvcurves = the kv_cv.R outer lambda rule); outer loop and fold assignment as tp_cnc_refit.R (= kv_cv.R).
# Outer folds only: no inner loop is needed for AUROC and paired differences (no nested selection between representations here).
#   FEAT=sub REP=r      Analysis A. Per outer fold: full L-CNV on the shipped matrix (must equal cv/preds/cfg_00), then refits on
#                       arms_cx (44 arms + cx, fixed) and top05/top10/top20 (top-k features of THAT fold's full L-CNV fit, ranked by
#                       |coefficient| at the chosen lambda, ties at 0 broken by entry order along the glmnet path; training rows only).
#   FEAT=pkg REP=r      the stored joint package block (must equal cv/preds/cfg_01).
#   FEAT=thin_XXX REP=r Analysis B: killcoyne_mm/two_papers_CND/thin/<FEAT>/cnv_block_C.csv (tp_cnd_thin.R MODE=assemble).
#   FEAT=ablate REP=r   Analysis C (ablation): full L-CNV again (abl_full, must equal cfg_00), abl_no3arms = every shipped feature overlapping
#                       hg19 arms 4q, 7p or 17p removed (the 3 arm columns and every 5-Mb tile overlapping them; cx kept), abl_arms_cx_no3 = 44 arms + cx
#                       minus the 4q, 7p and 17p arms.
# Blocks are used as stored (already standardised, as kv_cv.R). Row-level output -> killcoyne_mm/two_papers_CND/refit/<set>/outer_rep_XX.csv
# and refit/sel/sel_rep_XX.csv (cluster only).
source("/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/scripts/paper_plan/kv_common.R")
FEAT <- Sys.getenv("FEAT"); REP <- as.integer(Sys.getenv("REP")); E <- file.path(M, "two_papers_CND")
fout <- function(set) { d <- file.path(E, "refit", set); dir.create(d, recursive = TRUE, showWarnings = FALSE); file.path(d, sprintf("outer_rep_%02d.csv", REP)) }
sets <- if (FEAT == "sub") c("their", "arms_cx", "top05", "top10", "top20") else if (FEAT == "ablate") c("abl_full", "abl_no3arms", "abl_arms_cx_no3") else FEAT
if (all(file.exists(sapply(sets, fout)))) quit(save = "no")
RAW <- if (FEAT %in% c("sub", "their", "ablate")) readblock("cnv_their_C.csv") else if (FEAT == "pkg") readblock("cnv_pkg_C.csv") else readblock(file.path("two_papers_CND/thin", FEAT, "cnv_block_C.csv"))
strat_folds <- function(pp, yy, seed, nf) { pl <- tapply(yy, pp, max); Pp <- names(pl)[pl == 1]; Np <- names(pl)[pl == 0]; set.seed(seed); ord <- c(sample(Pp), sample(Np)); setNames(rep(1:nf, length.out = length(ord)), ord) }
fold_of <- strat_folds(pat, y, REP, 10)   # identical to kv_cv.R
fitpred <- function(cols, tr, te) { cv <- cvcurves(RAW[tr, cols, drop = FALSE], y[tr], pat[tr]); b <- as.numeric(as.matrix(coef(cv$fit, s = cv$class_min)))[-1]
  list(cv = cv, b = b, pred = predict(cv$fit, RAW[te, cols, drop = FALSE], s = cv$class_min, type = "response")[, 1]) }
if (FEAT %in% c("sub", "ablate")) { stopifnot(ncol(RAW) == 634, colnames(RAW)[634] == "cx"); ARMCX <- 590:634 }
if (FEAT == "ablate") {
  q <- do.call(rbind, strsplit(sub("-", ":", colnames(RAW)[-634]), ":")); ch <- q[, 1]; st <- as.numeric(q[, 2]); en <- as.numeric(q[, 3])
  A3 <- c("4:52700000-191154276", "7:1-58000000", "17:1-22200000"); stopifnot(all(A3 %in% colnames(RAW)[ARMCX]))   # hg19 arms 4q, 7p, 17p of the shipped matrix
  hit <- Reduce(`|`, lapply(A3, function(a) { z <- strsplit(sub("-", ":", a), ":")[[1]]; ch == z[1] & st <= as.numeric(z[3]) & en >= as.numeric(z[2]) }))
  KEEP1 <- c(which(!hit), 634); KEEP2 <- setdiff(ARMCX, which(hit)); stopifnot(length(KEEP2) == 42)
  message("ablate: removed ", sum(hit), " features (", sum(hit[ARMCX[-45]]), " arms, ", sum(hit[1:589]), " 5-Mb tiles); kept ", length(KEEP1), " and ", length(KEEP2))
}
O <- list(); S <- list()
for (k in 1:10) {
  te <- which(fold_of[pat] == k); tr <- which(fold_of[pat] != k); row <- function(set, p) data.frame(Sample = ids[te], Patient = pat[te], y = y[te], set = set, rep = REP, fold = k, pred = p$pred, lambda = p$cv$class_min, nonzero = sum(p$b != 0))
  full <- fitpred(seq_len(ncol(RAW)), tr, te); O[[length(O) + 1]] <- row(if (FEAT == "sub") "their" else if (FEAT == "ablate") "abl_full" else FEAT, full)
  if (FEAT == "ablate") { for (z in list(list("abl_no3arms", KEEP1), list("abl_arms_cx_no3", KEEP2))) O[[length(O) + 1]] <- cbind(row(z[[1]], fitpred(z[[2]], tr, te)), n_features = length(z[[2]])); next }
  if (FEAT != "sub") next
  S[[length(S) + 1]] <- data.frame(rep = REP, fold = k, set = "their_nonzero", feature = colnames(RAW)[full$b != 0], rank = NA, coef = full$b[full$b != 0])
  a <- fitpred(ARMCX, tr, te); O[[length(O) + 1]] <- row("arms_cx", a)
  S[[length(S) + 1]] <- data.frame(rep = REP, fold = k, set = "arms_cx_nonzero", feature = colnames(RAW)[ARMCX][a$b != 0], rank = NA, coef = a$b[a$b != 0])
  ent <- apply(as.matrix(full$cv$fit$beta) != 0, 1, function(v) if (any(v)) which(v)[1] else Inf)   # entry step along the path
  ord <- order(-abs(full$b), ent)
  for (kk in c(5, 10, 20)) {
    cols <- ord[1:kk]; p <- fitpred(cols, tr, te); nm <- sprintf("top%02d", kk); O[[length(O) + 1]] <- row(nm, p)
    S[[length(S) + 1]] <- data.frame(rep = REP, fold = k, set = nm, feature = colnames(RAW)[cols], rank = 1:kk, coef = p$b, full_nonzero = sum(full$b != 0))
  }
}
if (FEAT == "sub") { d <- file.path(E, "refit", "sel"); dir.create(d, recursive = TRUE, showWarnings = FALSE); write_csv(bind_rows(S), file.path(d, sprintf("sel_rep_%02d.csv", REP))) }
O <- bind_rows(O); for (s in sets) write_csv(O[O$set == s, ], fout(s))
message("TP CND REFIT DONE ", FEAT, " rep ", REP)
