library(checkmate)
library(mlr3)
# Load test scaffolding without helper_debugging.R temporarily
# See https://github.com/mlr-org/mlr3torch/issues/8
lapply(list.files(system.file("testthat", package = "mlr3"), pattern = "^helper.*\\.[rR]$", full.names = TRUE)[-2], source)
