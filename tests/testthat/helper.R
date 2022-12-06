library(checkmate)
library(mlr3)
library(mlr3misc)
# Load test scaffolding without helper_debugging.R temporarily
# See https://github.com/mlr-org/mlr3torch/issues/8

mlr_test_helpers = list.files(system.file("testthat", package = "mlr3"), pattern = "^helper.*\\.[rR]$", full.names = TRUE)
mlr_test_helpers = mlr_test_helpers[!grepl("helper\\_debugging\\.[rR]", mlr_test_helpers)]
lapply(mlr_test_helpers, source)

torch_test_helpers = list.files(system.file("testthat", package = "mlr3torch"), pattern = "^helper.*\\.[rR]$", full.names = TRUE)
torch_test_helpers = torch_test_helpers[!grepl("helper\\_debugging\\.[rR]", torch_test_helpers)]
lapply(torch_test_helpers, source)

mlr_tasks$add("test_imagenet", load_task_test_imagenet)

rm(mlr_test_helpers)
