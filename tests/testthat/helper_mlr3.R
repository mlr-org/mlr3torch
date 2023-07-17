# load mlr3 helper files

library("mlr3")

mlr_helpers = list.files(system.file("testthat", package = "mlr3"), pattern = "^helper.*\\.[rR]", full.names = TRUE)
# torch assumes that one can access null fields of R6 instances
mlr_helpers = mlr_helpers[!grepl("helper_debugging", mlr_helpers)]

# this causes the
lapply(mlr_helpers, FUN = source)
