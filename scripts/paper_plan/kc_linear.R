# Killcoyne checks, items 2 and 4 (docs/paper_plan_killcoyne_checks.md @ 828ebf7). Same machinery as km_linear.R (glmnet alpha 0.9,
# standardize = FALSE, LOPO over set C, 10 x 5 patient-grouped CV with set.seed(r), rules class_min / dev_min / global).
# cfg 0-2: image block z-scored on the training patients of each fold (fold-honest); cfg 3-7: pathology grade arms (whole-set scaling).
suppressMessages({library(glmnet); library(readr); library(dplyr)})
K <- "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/feasibility/paper_plan/killcoyne"; M <- paste0(K, "_mm"); OD <- file.path(M, "checks", "linear"); dir.create(OD, recursive = TRUE, showWarnings = FALSE)
grid <- data.frame(model = c("img_fh", "early_fh", "early_fh", "grade", "cnvgrade", "cnvgrade", "earlygrade", "earlygrade"), cnvsrc = c("none", "their", "pkg", "none", "their", "pkg", "their", "pkg"), stringsAsFactors = FALSE)
cfg <- as.integer(Sys.getenv("CFG_ID", "0")); ch <- as.integer(Sys.getenv("CHUNK_ID", "0")); nch <- as.integer(Sys.getenv("N_CHUNKS", "1")); g <- grid[cfg + 1, ]
outf <- file.path(OD, sprintf("cfg_%02d_chunk_%02d.csv", cfg, ch)); if (file.exists(outf)) quit(save = "no")
C <- read_csv(file.path(M, "set_C.csv"), col_types = cols(.default = "c")); ids <- C$Sample; y <- as.integer(C$Status == "P"); pat <- C$Patient; n <- length(ids)
zs <- function(Mx, ref) { mu <- colMeans(Mx[ref, , drop = FALSE]); sdv <- apply(Mx[ref, , drop = FALSE], 2, sd); Z <- sweep(sweep(Mx, 2, mu), 2, ifelse(sdv > 0, sdv, 1), "/"); Z[, sdv == 0] <- 0; Z }
readblock <- function(f) { R <- read_csv(file.path(M, f), col_types = cols(sample_id = "c", .default = "d")); X <- as.matrix(R[, -1]); rownames(X) <- R$sample_id; X[ids, , drop = FALSE] }
CN <- if (g$cnvsrc != "none") readblock(sprintf("cnv_%s_C.csv", g$cnvsrc)) else NULL   # exported by km_linear.R EXPORT=1 (their: as shipped; pkg: plan B construction over C)
IMraw <- NULL; if (g$model %in% c("img_fh", "early_fh", "earlygrade")) { R <- read_csv(file.path(M, "img_mean.csv"), col_types = cols(Sample = "c", .default = "d")); IMraw <- as.matrix(R[, -1]); rownames(IMraw) <- R$Sample; IMraw <- IMraw[ids, ] }
gr <- c(NDBE = 0, ID = 1, LGD = 2, HGD = 3, IMC = 4)[C$Pathology]; stopifnot(!any(is.na(gr))); GR <- matrix((gr - mean(gr)) / sd(gr), ncol = 1, dimnames = list(ids, "grade"))
features <- function(tr) switch(g$model,
  img_fh = zs(IMraw, tr), early_fh = cbind(CN, zs(IMraw, tr)),
  grade = cbind(GR, pad = 0), cnvgrade = cbind(CN, GR), earlygrade = cbind(CN, zs(IMraw, seq_len(n)), GR))   # grade arms: whole-set image scaling, as the primary multimodal arms
shuffle_groups <- function(pp, r) { up <- unique(pp); set.seed(r); sp <- sample(up); setNames(rep(1:5, length.out = length(sp)), sp) }
cvcurves <- function(X, yy, pp) {
  fit <- glmnet(X, yy, family = "binomial", alpha = 0.9, standardize = FALSE); lam <- fit$lambda; cls <- dev <- matrix(NA, 10, length(lam))
  for (r in 1:10) {
    fo <- shuffle_groups(pp, r)[pp]; fe <- fd <- matrix(NA, 5, length(lam)); w <- numeric(5)
    for (k in 1:5) { te <- fo == k; f <- glmnet(X[!te, ], yy[!te], family = "binomial", alpha = 0.9, lambda = lam, standardize = FALSE); pr <- pmin(pmax(predict(f, X[te, , drop = FALSE], type = "response"), 1e-5), 1 - 1e-5)
      if (ncol(pr) < length(lam)) pr <- cbind(pr, matrix(pr[, ncol(pr)], nrow(pr), length(lam) - ncol(pr)))
      fe[k, ] <- colMeans((pr > 0.5) != yy[te]); fd[k, ] <- colMeans(-2 * (yy[te] * log(pr) + (1 - yy[te]) * log(1 - pr))); w[k] <- sum(te) }
    cls[r, ] <- colSums(fe * w) / sum(w); dev[r, ] <- colSums(fd * w) / sum(w) }
  cm <- colMeans(cls); dm <- colMeans(dev); list(fit = fit, class_min = max(lam[cm <= min(cm)]), dev_min = lam[which.min(dm)]) }
message("cfg ", cfg, " ", g$model, " ", g$cnvsrc)
gl <- cvcurves(features(seq_len(n)), y, pat)$class_min
up <- sort(unique(pat)); mine <- up[(seq_along(up) - 1) %% nch == ch]; out <- list()
for (p in mine) {
  te <- pat == p; tr <- which(!te); X <- features(tr); cv <- cvcurves(X[tr, ], y[tr], pat[tr])
  lams <- c(class_min = cv$class_min, dev_min = cv$dev_min, global = gl); pr <- predict(cv$fit, X[te, , drop = FALSE], s = lams, type = "response")
  for (j in seq_along(lams)) out[[length(out) + 1]] <- data.frame(Sample = ids[te], Patient = p, y = y[te], cfg = cfg, model = g$model, cnvsrc = g$cnvsrc, rule = names(lams)[j], lambda = lams[j], pred = pr[, j])
}
write_csv(bind_rows(out), outf); message("CHECKS LINEAR CHUNK DONE cfg ", cfg, " chunk ", ch)
