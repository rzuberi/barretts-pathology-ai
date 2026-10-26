# Two papers, agent P1A, item 1 (docs/two_papers_P1A.md): Killcoyne's model-P recipe refitted inside the stored 10 x 10 patient-grouped
# outer folds of kv_cv.R (same fold assignment: set.seed(REP); progressor patients shuffled, then non-progressors; cyclic over 10 folds).
# Recipe (Killcoyne 2020; docs/paper_plan_killcoyne_reconcile.md step table): features = their shipped 773 x 634 matrix (QDNAseq 50 kb,
# 5-Mb tiles, arm-adjusted tiles, arms, cx; standardised over their whole cohort) or the same pipeline run on our hg38 counts through the
# package ('pkg', standardised over the 773 published samples); glmnet alpha 0.9, standardize = FALSE, nlambda = NLAM (1000 as shipped);
# lambda from 10 x 5-fold patient-grouped CV within the training patients (kr_lopo.R cvcurves, copied), rules class_min, class_1se,
# dev_min, dev_1se, plus 'published' (the shipped model's lambda, chosen on all data: not fold-honest, labelled optimistic).
# Training rows: UNIV = s773 -> every published sample of the training patients plus all samples of the 8 patients without a slide (never
#   evaluated, so always training); UNIV = C -> the set C rows of the training patients (the rows L-CNV is trained on).
# Test rows: the held-out patients' set C rows. Output: killcoyne_mm/two_papers_P1A/prefit/<FEAT>_<UNIV>_nl<NLAM>_rep_RR_fold_KK.csv
# MODE=boot DRAW FOLD: the same recipe inside one patient-bootstrap draw of review_round2/f3/draws.csv (rv2_refit.R MODE=f3 conventions:
#   drawn patients' rows replicated by multiplicity, strat_folds(.., DRAW, 10) on patient IDs, copies share every split); outer predictions only.
suppressMessages({library(glmnet); library(readr); library(dplyr)})
TH <- "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis"; K <- file.path(TH, "feasibility/paper_plan/killcoyne"); M <- paste0(K, "_mm")
FEAT <- Sys.getenv("FEAT", "their"); UNIV <- Sys.getenv("UNIV", "s773"); NLAM <- as.integer(Sys.getenv("NLAM", "1000")); MODE <- Sys.getenv("MODE", "outer")
REP <- as.integer(Sys.getenv("REP", "1")); FOLD <- as.integer(Sys.getenv("FOLD", "1")); DRAW <- as.integer(Sys.getenv("DRAW", "0"))
OD <- file.path(M, "two_papers_P1A", if (MODE == "boot") "prefit_boot" else "prefit"); dir.create(OD, recursive = TRUE, showWarnings = FALSE)
outf <- if (MODE == "boot") file.path(OD, sprintf("%s_%s_nl%d_draw_%03d_fold_%02d.csv", FEAT, UNIV, NLAM, DRAW, FOLD)) else file.path(OD, sprintf("%s_%s_nl%d_rep_%02d_fold_%02d.csv", FEAT, UNIV, NLAM, REP, FOLD))
if (file.exists(outf)) quit(save = "no")
C <- read_csv(file.path(M, "set_C.csv"), col_types = cols(.default = "c")); cids <- C$Sample; cy <- as.integer(C$Status == "P"); cpat <- C$Patient
smp <- read_csv(file.path(K, "kr_samples.csv"), col_types = cols(.default = "c")); pub_lambda <- as.numeric(readLines(file.path(K, "be_model_lambda.txt")))
stopifnot(all(cids %in% smp$Sample), all(smp$Patient[match(cids, smp$Sample)] == cpat))
uids <- if (UNIV == "s773") smp$Sample else cids
zs <- function(Mx, ref) { mu <- colMeans(Mx[ref, , drop = FALSE]); sdv <- apply(Mx[ref, , drop = FALSE], 2, sd); Z <- sweep(sweep(Mx, 2, mu), 2, ifelse(sdv > 0, sdv, 1), "/"); Z[, sdv == 0] <- 0; Z }
if (FEAT == "their") {
  R <- read_csv(file.path(K, "their_features.csv"), col_types = cols(Sample = "c", .default = "d")); X0 <- as.matrix(R[, -1]); rownames(X0) <- R$Sample
  uids <- intersect(uids, rownames(X0)); X0 <- X0[uids, ]                     # shipped matrix used as shipped (already standardised over 773)
} else if (FEAT == "pkg" && UNIV == "C") {
  R <- read_csv(file.path(M, "cnv_pkg_C.csv"), col_types = cols(sample_id = "c", .default = "d")); X0 <- as.matrix(R[, -1]); rownames(X0) <- R$sample_id; X0 <- X0[uids, ]   # exactly the stored L-CNV package block
} else {
  P <- readRDS(file.path(K, "pkg_tiles.rds")); uids <- intersect(uids, rownames(P$seg))
  loc <- function(n) { a <- do.call(rbind, strsplit(sub("-", ":", n), ":")); data.frame(chr = a[, 1], s = as.numeric(a[, 2]), e = as.numeric(a[, 3])) }
  lt <- loc(colnames(P$seg)); la <- loc(colnames(P$arm)); W <- matrix(0, ncol(P$arm), ncol(P$seg)); for (i in seq_len(nrow(la))) W[i, lt$chr == la$chr[i] & lt$s <= la$e[i] & lt$e >= la$s[i]] <- 1
  ref <- seq_along(uids); S <- zs(P$seg[uids, ], ref); A <- zs(P$arm[uids, ], ref); cx <- apply(S, 1, function(x) sum(x >= mean(x) + 2 * sd(x) | x <= mean(x) - 2 * sd(x))); cx <- (cx - mean(cx)) / sd(cx)
  X0 <- cbind(S - A %*% W, A, cx = cx); X0[!is.finite(X0)] <- 0                # kr_lopo.R 'pkg', cohort standardisation over the 773
}
stopifnot(all(cids %in% uids)); s <- smp[match(uids, smp$Sample), ]; uy <- as.integer(s$Status == "P"); upat <- s$Patient
message("FEAT ", FEAT, " UNIV ", UNIV, " rows ", length(uids), " patients ", length(unique(upat)), " cols ", ncol(X0), " NLAM ", NLAM)
cvcurves <- function(X, yy, pp) {   # kr_lopo.R cvcurves (alpha 0.9, standardize FALSE), copied
  fit <- glmnet(X, yy, family = "binomial", alpha = 0.9, standardize = FALSE, nlambda = NLAM); lam <- fit$lambda; up <- unique(pp)
  cls <- dev <- se <- sed <- matrix(NA, 10, length(lam))
  for (r in 1:10) {
    set.seed(r); sp <- sample(up); fo <- setNames(rep(1:5, length.out = length(sp)), sp)[pp]; fe <- matrix(NA, 5, length(lam)); fd <- matrix(NA, 5, length(lam)); w <- numeric(5)
    for (k in 1:5) { te <- fo == k; f <- glmnet(X[!te, ], yy[!te], family = "binomial", alpha = 0.9, lambda = lam, standardize = FALSE); pr <- predict(f, X[te, , drop = FALSE], type = "response")
      if (ncol(pr) < length(lam)) pr <- cbind(pr, matrix(pr[, ncol(pr)], nrow(pr), length(lam) - ncol(pr)))
      fe[k, ] <- colMeans((pr > 0.5) != yy[te]); pc <- pmin(pmax(pr, 1e-5), 1 - 1e-5); fd[k, ] <- colMeans(-2 * (yy[te] * log(pc) + (1 - yy[te]) * log(1 - pc))); w[k] <- sum(te) }
    cls[r, ] <- colSums(fe * w) / sum(w); dev[r, ] <- colSums(fd * w) / sum(w); se[r, ] <- sqrt(colSums(w * sweep(fe, 2, cls[r, ])^2) / sum(w) / 4); sed[r, ] <- sqrt(colSums(w * sweep(fd, 2, dev[r, ])^2) / sum(w) / 4)
  }
  cm <- colMeans(cls); dm <- colMeans(dev); sm <- colMeans(se); sdm <- colMeans(sed); imin <- which(cm <= min(cm)); dmin <- which.min(dm)
  list(fit = fit, class_min = max(lam[imin]), class_1se = max(lam[cm <= cm[max(imin)] + sm[max(imin)]]), dev_min = lam[dmin], dev_1se = max(lam[dm <= dm[dmin] + sdm[dmin]]))
}
strat_folds <- function(pp, yy, seed, nf) { pl <- tapply(yy, pp, max); Pp <- names(pl)[pl == 1]; Np <- names(pl)[pl == 0]; set.seed(seed); ord <- c(sample(Pp), sample(Np)); setNames(rep(1:nf, length.out = length(ord)), ord) }
nnz <- function(fit, s) sum(as.numeric(as.matrix(coef(fit, s = s)))[-1] != 0)
emit <- function(cv, Xte, meta) {
  lams <- c(class_min = cv$class_min, class_1se = cv$class_1se, dev_min = cv$dev_min, dev_1se = cv$dev_1se, published = pub_lambda)
  pr <- predict(cv$fit, Xte, s = lams, type = "response"); bind_rows(lapply(seq_along(lams), function(j) data.frame(meta, rule = names(lams)[j], lambda = lams[j], nnz = nnz(cv$fit, lams[j]), pred = pr[, j])))
}
if (MODE == "outer") {
  fold_of <- strat_folds(cpat, cy, REP, 10)                                    # == kv_cv.R fold assignment over the 80 set C patients
  tep <- names(fold_of)[fold_of == FOLD]; tr <- which(!upat %in% tep); te <- match(cids[cpat %in% tep], uids)
  cv <- cvcurves(X0[tr, ], uy[tr], upat[tr])
  out <- emit(cv, X0[te, , drop = FALSE], data.frame(Sample = uids[te], Patient = upat[te], y = uy[te], feat = FEAT, univ = UNIV, nlam = NLAM, rep = REP, fold = FOLD, n_train = length(tr), n_train_patients = length(unique(upat[tr]))))
} else if (MODE == "boot") {
  DR <- read_csv(file.path(M, "review_round2", "f3", "draws.csv"), col_types = cols(draw = "i", Patient = "c", mult = "i")); d <- DR[DR$draw == DRAW, ]; stopifnot(nrow(d) > 0)
  extra <- setdiff(unique(upat), unique(cpat))                                 # patients without a slide (s773 only): kept once in every training set
  ri <- integer(0); cp <- integer(0); key <- character(0)
  for (i in seq_len(nrow(d))) { rows <- which(upat == d$Patient[i]); for (cc in seq_len(d$mult[i])) { ri <- c(ri, rows); cp <- c(cp, rep(cc, length(rows))) } }
  dp <- upat[ri]; fo3 <- strat_folds(dp, uy[ri], DRAW, 10)                       # identical to rv2_refit.R MODE=f3 (pre-event population draws over set C patients)
  te_i <- ri[fo3[dp] == FOLD & uids[ri] %in% cids]; te_cp <- cp[fo3[dp] == FOLD & uids[ri] %in% cids]
  tr_i <- c(ri[fo3[dp] != FOLD], which(upat %in% extra))
  cv <- cvcurves(X0[tr_i, ], uy[tr_i], upat[tr_i])
  out <- emit(cv, X0[te_i, , drop = FALSE], data.frame(Sample = uids[te_i], Patient = upat[te_i], copy = te_cp, y = uy[te_i], feat = FEAT, univ = UNIV, nlam = NLAM, draw = DRAW, fold = FOLD, n_train = length(tr_i)))
} else stop("unknown MODE")
write_csv(out, outf); message("TP P1A PREFIT DONE ", basename(outf))
