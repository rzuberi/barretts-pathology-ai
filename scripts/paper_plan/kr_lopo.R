# Killcoyne reconciliation plans B and C (docs/paper_plan_killcoyne_reconcile.md @ a655e08). One configuration (CFG_ID) of the grid
# set x features x standardisation x alpha; leave-one-patient-out over the patients of CHUNK_ID (of N_CHUNKS). For every left-out patient:
# 10 x 5-fold patient-grouped CV on the remaining patients (fold patients shuffled with set.seed(r), r = 1..10) over the lambda path of the
# training fit; lambda rules read off the same curves: class min, class 1-s.e., deviance min; plus 'global' (class-min lambda from the same CV
# on the whole set, reused for every left-out patient) and 'published' (be_model lambda). Label = sheet Status (P = 1), per sample.
# Plan B = cfg (s773, pkg, cohort, 0.9) with rule class_min. Output rows: feasibility/paper_plan/killcoyne/lopo/cfg_XX_chunk_Y.csv.
suppressMessages({library(glmnet); library(readr); library(dplyr)})
O <- "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/feasibility/paper_plan/killcoyne"
cfg <- as.integer(Sys.getenv("CFG_ID", "0")); ch <- as.integer(Sys.getenv("CHUNK_ID", "0")); nch <- as.integer(Sys.getenv("N_CHUNKS", "1"))
# rows 0-31: pre-specified grid (release features exist only for release rows, so rel x s773/s711 is NOT AVAILABLE and omitted); glmnet defaults.
pre <- expand.grid(alpha = c(0.9, 0.5, 0.7, 1.0), std = c("cohort", "train"), feat = c("pkg", "rel"), set = c("s773", "s711", "rel504"), stringsAsFactors = FALSE)
pre <- pre[!(pre$feat == "rel" & pre$set != "rel504"), ]; pre$stdz <- TRUE; pre$nlam <- 100; pre$tag <- "prespec"
# rows 32+: added after the pre-specification, informed by the shipped model (fit call glmnet(..., standardize = F, nlambda = nl) with a
# 967-value path; per-patient LOO coefficient tables headed 'non.zero.coef(fitLOO, cv$lambda.1se)'); feature source 'their' = the shipped
# 773 x 634 training matrix, rows aligned to our samples by kr_align.py, used as shipped (already whole-cohort standardised).
post <- data.frame(alpha = 0.9, std = "cohort", feat = c("their", "their", "their", "their", "their", "pkg", "pkg", "rel"),
                   set = c("s773", "s773", "s711", "rel504", "s773", "s773", "rel504", "rel504"), stdz = c(FALSE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE),
                   nlam = c(100, 100, 100, 100, 1000, 100, 100, 100), tag = "post_hoc", stringsAsFactors = FALSE)
grid <- rbind(pre, post); rownames(grid) <- NULL
if (Sys.getenv("PRINT_GRID") == "1") { write.csv(cbind(cfg = seq_len(nrow(grid)) - 1, grid), file.path("/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/feasibility/paper_plan/killcoyne", "grid.csv"), row.names = FALSE); quit(save = "no") }
g <- grid[cfg + 1, ]; message("cfg ", cfg, ": ", paste(names(g), g, collapse = " "))
dir.create(file.path(O, "lopo"), showWarnings = FALSE); outf <- file.path(O, "lopo", sprintf("cfg_%02d_chunk_%d.csv", cfg, ch)); if (file.exists(outf)) quit(save = "no")
smp <- read_csv(file.path(O, "kr_samples.csv"), col_types = cols(.default = "c"))
pub_lambda <- as.numeric(readLines(file.path(O, "be_model_lambda.txt")))
rel_rows <- read_csv(file.path(O, "rel_rows.csv"), col_types = cols(.default = "c"))$Sample
if (g$feat == "pkg") {
  P <- readRDS(file.path(O, "pkg_tiles.rds")); avail <- rownames(P$seg)
  # arm-by-tile incidence: subtractArms subtracts every overlapping arm from a tile
  loc <- function(n) { a <- do.call(rbind, strsplit(sub("-", ":", n), ":")); data.frame(chr = a[, 1], s = as.numeric(a[, 2]), e = as.numeric(a[, 3])) }
  lt <- loc(colnames(P$seg)); la <- loc(colnames(P$arm)); W <- matrix(0, ncol(P$arm), ncol(P$seg))
  for (i in seq_len(nrow(la))) W[i, lt$chr == la$chr[i] & lt$s <= la$e[i] & lt$e >= la$s[i]] <- 1
} else { R <- read_csv(file.path(O, if (g$feat == "rel") "rel_features.csv" else "their_features.csv"), col_types = cols(Sample = "c", .default = "d")); RM <- as.matrix(R[, -1]); rownames(RM) <- R$Sample; avail <- R$Sample }
ids <- switch(g$set, s773 = smp$Sample, s711 = smp$Sample[!smp$Pathology %in% c("HGD", "IMC")], rel504 = rel_rows)
ids <- intersect(ids, avail); s <- smp[match(ids, smp$Sample), ]; y <- as.integer(s$Status == "P"); pat <- s$Patient; n <- length(ids)
message("rows ", n, " patients ", length(unique(pat)), " P rows ", sum(y))
zs <- function(M, ref) { mu <- colMeans(M[ref, , drop = FALSE]); sdv <- apply(M[ref, , drop = FALSE], 2, sd); Z <- sweep(sweep(M, 2, mu), 2, ifelse(sdv > 0, sdv, 1), "/"); Z[, sdv == 0] <- 0; Z }
cxs <- function(Z) apply(Z, 1, function(x) sum(x >= mean(x) + 2 * sd(x) | x <= mean(x) - 2 * sd(x)))
features <- function(ref) {   # ref = row indices whose mean/sd standardise the data (package order: scale tiles, cx on scaled tiles, scale cx, subtract arms)
  if (g$feat == "pkg") { S <- zs(P$seg[ids, ], ref); A <- zs(P$arm[ids, ], ref); cx <- cxs(S); cx <- (cx - mean(cx[ref])) / sd(cx[ref]); X <- cbind(S - A %*% W, A, cx = cx)
  } else if (g$feat == "their") { X <- RM[ids, ]   # shipped matrix, already standardised over their whole cohort
  } else { X <- zs(RM[ids, ], ref) }
  X[!is.finite(X)] <- 0; X }
cvcurves <- function(X, yy, pp) {   # 10 x 5-fold patient-grouped CV over the path of the full fit on (X, yy)
  fit <- glmnet(X, yy, family = "binomial", alpha = g$alpha, standardize = g$stdz, nlambda = g$nlam); lam <- fit$lambda; up <- unique(pp)
  cls <- dev <- se <- sed <- matrix(NA, 10, length(lam))
  for (r in 1:10) {
    set.seed(r); sp <- sample(up); fo <- setNames(rep(1:5, length.out = length(sp)), sp)[pp]; fe <- matrix(NA, 5, length(lam)); fd <- matrix(NA, 5, length(lam)); w <- numeric(5)
    for (k in 1:5) { te <- fo == k; f <- glmnet(X[!te, ], yy[!te], family = "binomial", alpha = g$alpha, lambda = lam, standardize = g$stdz); pr <- predict(f, X[te, , drop = FALSE], type = "response"); pr <- pmin(pmax(pr, 1e-5), 1 - 1e-5)
      if (ncol(pr) < length(lam)) pr <- cbind(pr, matrix(pr[, ncol(pr)], nrow(pr), length(lam) - ncol(pr)))
      fe[k, ] <- colMeans((pr > 0.5) != yy[te]); fd[k, ] <- colMeans(-2 * (yy[te] * log(pr) + (1 - yy[te]) * log(1 - pr))); w[k] <- sum(te) }
    cls[r, ] <- colSums(fe * w) / sum(w); dev[r, ] <- colSums(fd * w) / sum(w); se[r, ] <- sqrt(colSums(w * sweep(fe, 2, cls[r, ])^2) / sum(w) / 4); sed[r, ] <- sqrt(colSums(w * sweep(fd, 2, dev[r, ])^2) / sum(w) / 4)
  }
  cm <- colMeans(cls); dm <- colMeans(dev); sm <- colMeans(se); sdm <- colMeans(sed); imin <- which(cm <= min(cm)); dmin <- which.min(dm)
  list(fit = fit, lam = lam, class_min = max(lam[imin]), class_1se = max(lam[cm <= cm[max(imin)] + sm[max(imin)]]), dev_min = lam[dmin], dev_1se = max(lam[dm <= dm[dmin] + sdm[dmin]]))
}
Xc <- features(seq_len(n)); gl <- cvcurves(Xc, y, pat)$class_min   # 'global' rule: lambda chosen once on the whole set
up <- sort(unique(pat)); mine <- up[(seq_along(up) - 1) %% nch == ch]; out <- list()
for (p in mine) {
  te <- pat == p; tr <- which(!te); X <- if (g$std == "cohort") Xc else features(tr)
  cv <- cvcurves(X[tr, ], y[tr], pat[tr]); lams <- c(class_min = cv$class_min, class_1se = cv$class_1se, dev_min = cv$dev_min, dev_1se = cv$dev_1se, global = gl, published = pub_lambda)   # dev_1se added post hoc
  pr <- predict(cv$fit, X[te, , drop = FALSE], s = lams, type = "response")
  for (j in seq_along(lams)) out[[length(out) + 1]] <- data.frame(Sample = ids[te], Patient = p, y = y[te], cfg = cfg, tag = g$tag, rule = names(lams)[j], lambda = lams[j], pred = pr[, j])
}
write_csv(bind_rows(out), outf); message("LOPO CHUNK DONE cfg ", cfg, " chunk ", ch, " patients ", length(mine))
