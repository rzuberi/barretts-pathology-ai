# Two papers, agent CNC: install ACE, HMMcopy, ichorCNA and CINSignatureQuantification into a separate library (envs/cnc_rlib) used with the
# killcoyne_r R; the killcoyne_r library itself is not modified.
L <- "/mnt/scratche/slow/fmlab/zuberi01/envs/cnc_rlib"; dir.create(L, showWarnings = FALSE); .libPaths(c(L, .libPaths()))
options(repos = c(CRAN = "https://cloud.r-project.org"), Ncpus = 4)
if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager", lib = L)
if (!requireNamespace("remotes", quietly = TRUE)) install.packages("remotes", lib = L)
for (p in c("ACE", "HMMcopy")) if (!requireNamespace(p, quietly = TRUE)) BiocManager::install(p, lib = L, update = FALSE, ask = FALSE)
if (!requireNamespace("ichorCNA", quietly = TRUE)) remotes::install_github("broadinstitute/ichorCNA", lib = L, upgrade = "never", dependencies = TRUE)
if (!requireNamespace("CINSignatureQuantification", quietly = TRUE)) remotes::install_github("markowetzlab/CINSignatureQuantification", lib = L, upgrade = "never", dependencies = TRUE)
for (p in c("ACE", "HMMcopy", "ichorCNA", "CINSignatureQuantification")) cat(p, requireNamespace(p, quietly = TRUE), as.character(tryCatch(packageVersion(p, lib.loc = L), error = function(e) NA)), "\n")
