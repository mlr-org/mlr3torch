devtools::install_github("mlr-org/mlr3torch")
devtools::install_github("mlr-org/mlr3tuning@fix/int-tune-trafo")

# Package names
packages = c("here", "mlr3oml", "tidytable", "mlr3", "mlr3learners", "mlr3tuning", "mlr3mbo", "bbotk", "bench", "data.table")

# Install packages not yet installed
installed_packages = packages %in% rownames(installed.packages())
if (any(installed_packages == FALSE)) {
  install.packages(packages[!installed_packages], repos = "https://ftp.fau.de/cran/")
}