# Killcoyne reconciliation plan A, step 2 (docs/paper_plan_killcoyne_reconcile.md @ a655e08): gather the per-patient tiles from
# kr_package.R; score every segmented sample with the packaged frozen model (be_model: its tile/arm/cx means and s.d., its lambda).
# The model's 589 hg19 tiles are matched to our 587 hg38 tiles by nearest midpoint on the same chromosome (the package's own
# tileSamples() indexes tiles by position, which would misalign), arms by chromosome and p/q. Also writes our features built the
# model's way with whole-cohort standardisation, and the model's shipped training matrix, for row alignment (kr_align.py).
suppressMessages({library(BarrettsProgressionRisk); library(glmnet); library(readr); library(dplyr)})
O <- "/mnt/scratche/slow/fmlab/zuberi01/phd/thesis/feasibility/paper_plan/killcoyne"; m <- BarrettsProgressionRisk:::be_model
fs <- sort(list.files(file.path(O, "pkg"), "^pat_.*rds$", full.names = TRUE)); L <- lapply(fs, readRDS); ok <- !sapply(L, `[[`, "error")
cat("patient files", length(L), "segmented ok", sum(ok), "\n"); L <- L[ok]
mp <- bind_rows(lapply(L, `[[`, "map")); qc <- bind_rows(lapply(L, function(x) x$qc %>% mutate(samplename = as.character(samplename))))
seg <- do.call(rbind, lapply(L, `[[`, "seg")); arm <- do.call(rbind, lapply(L, `[[`, "arm"))
rownames(seg) <- mp$Sample[match(rownames(seg), mp$sid)]; rownames(arm) <- mp$Sample[match(rownames(arm), mp$sid)]
qc$Sample <- mp$Sample[match(qc$samplename, mp$sid)]
cat("segmented samples", nrow(seg), "QC pass", sum(qc$Pass), "tiles", ncol(seg), "arms", ncol(arm), "\n")
saveRDS(list(seg = seg, arm = arm, qc = qc), file.path(O, "pkg_tiles.rds"))
loc <- function(n) { a <- do.call(rbind, strsplit(sub("-", ":", n), ":")); data.frame(chr = a[, 1], s = as.numeric(a[, 2]), e = as.numeric(a[, 3])) }
mt <- loc(names(m$tile.mean)); ot <- loc(colnames(seg)); ma <- loc(names(m$arms.mean)); oa <- loc(colnames(arm))
ti <- sapply(seq_len(nrow(mt)), function(i) { w <- which(ot$chr == mt$chr[i]); w[which.min(abs((ot$s[w] + ot$e[w]) / 2 - (mt$s[i] + mt$e[i]) / 2))] })
ai <- sapply(seq_len(nrow(ma)), function(i) { w <- which(oa$chr == ma$chr[i]); w[if (ma$s[i] == 1) which.min(oa$s[w]) else which.max(oa$s[w])] })
cat("model tiles", nrow(mt), "mapped to distinct hg38 tiles", length(unique(ti)), "; arms", nrow(ma), "distinct", length(unique(ai)), "\n")
Wm <- matrix(0, nrow(ma), nrow(mt)); for (i in seq_len(nrow(ma))) Wm[i, mt$chr == ma$chr[i] & mt$s <= ma$e[i] & mt$e >= ma$s[i]] <- 1   # model-coordinate arm/tile overlaps (subtractArms)
S0 <- seg[, ti]; A0 <- arm[, ai]; colnames(S0) <- names(m$tile.mean); colnames(A0) <- names(m$arms.mean)
build <- function(mu_s, sd_s, mu_a, sd_a, cxm, cxsd) {
  S <- sweep(sweep(S0, 2, mu_s), 2, sd_s, "/"); A <- sweep(sweep(A0, 2, mu_a), 2, sd_a, "/")
  cx <- apply(S, 1, function(x) sum(x >= mean(x) + 2 * sd(x) | x <= mean(x) - 2 * sd(x))); cx <- (cx - cxm) / cxsd
  X <- cbind(S - A %*% Wm, A, cx = cx); colnames(X) <- c(names(m$tile.mean), names(m$arms.mean), "cx"); X[!is.finite(X)] <- 0; X }
# plan A: the model's own scaling constants
XA <- build(m$tile.mean, m$tile.sd, m$arms.mean, m$arms.sd, m$cx.mean, m$cx.sd)
cf <- rownames(coef(m$fit)); stopifnot(all(cf[-1] == colnames(XA)))
pA <- predict(m$fit, newx = XA, s = m$lambda, type = "response")[, 1]
write_csv(tibble(Sample = rownames(XA), probA = pA, pass = qc$Pass[match(rownames(XA), qc$Sample)], varMAD = qc$varMAD_median[match(rownames(XA), qc$Sample)]), file.path(O, "kr_A_pred.csv"))
# our features the model's way but standardised over our own 773 (for aligning the shipped training rows)
cxr <- apply(sweep(sweep(S0, 2, colMeans(S0)), 2, apply(S0, 2, sd), "/"), 1, function(x) sum(x >= mean(x) + 2 * sd(x) | x <= mean(x) - 2 * sd(x)))
XC <- build(colMeans(S0), apply(S0, 2, sd), colMeans(A0), apply(A0, 2, sd), mean(cxr), sd(cxr))
write_csv(as_tibble(XC) %>% mutate(Sample = rownames(XC), .before = 1), file.path(O, "ours_modelway_cohortstd.csv"))
write_csv(as_tibble(m$fit.data), file.path(O, "fitdata.csv"))
write_csv(tibble(row = seq_len(nrow(m$fit.data)), insample = predict(m$fit, newx = m$fit.data, s = m$lambda, type = "response")[, 1]), file.path(O, "be_model_insample.csv"))
cat("ASSEMBLE DONE\n")
