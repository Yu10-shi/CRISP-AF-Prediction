# R package dependencies for the Shiny app.
# Verified working with R 4.2.2 on 2026-06-21.
# Run once on the target machine:  Rscript install_r_packages.R
#
# IMPORTANT: reticulate MUST be >= 1.38. Older reticulate (e.g. 1.27) cannot
# talk to NumPy 2.x and fails at prediction time with:
#   "Required version of NumPy not available: incompatible NumPy binary version"
# We use NumPy 2.2.6, so a recent reticulate is required.

required <- c(
  shiny      = "1.7.4",
  shinyjs    = "2.1.0",
  DT         = "0.27",
  reticulate = "1.46.0",   # any >= 1.38 is fine (NumPy 2.x support)
  jsonlite   = "1.8.4",
  ggplot2    = "3.5.1",
  gridExtra  = "2.3",
  scales     = "1.3.0"
)

repos <- "https://cloud.r-project.org"
installed <- rownames(installed.packages())

for (pkg in names(required)) {
  if (!(pkg %in% installed)) {
    message(sprintf("Installing %s ...", pkg))
    install.packages(pkg, repos = repos)
  } else {
    message(sprintf("%s already installed (%s)", pkg, packageVersion(pkg)))
  }
}

# Enforce the reticulate floor
if (packageVersion("reticulate") < "1.38") {
  message("reticulate is older than 1.38 — upgrading for NumPy 2.x support ...")
  install.packages("reticulate", repos = repos)
}

cat("\nDone. Versions:\n")
for (pkg in names(required)) {
  cat(sprintf("  %-12s %s\n", pkg, tryCatch(as.character(packageVersion(pkg)),
                                            error = function(e) "NOT INSTALLED")))
}
