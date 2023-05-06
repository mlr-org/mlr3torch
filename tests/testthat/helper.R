library(checkmate)
library(mlr3)
library(mlr3misc)
# Load test scaffolding without helper_debugging.R temporarily
# See https://github.com/mlr-org/mlr3torch/issues/8

mlr_test_helpers = list.files(system.file("testthat", package = "mlr3"), pattern = "^helper.*\\.[rR]$",
  full.names = TRUE)
mlr_test_helpers = mlr_test_helpers[!grepl("helper\\_debugging\\.[rR]", mlr_test_helpers)]
lapply(mlr_test_helpers, source)

torch_test_helpers = list.files(system.file("testthat", package = "mlr3torch"), pattern = "^helper.*\\.[rR]$",
  full.names = TRUE)
torch_test_helpers = torch_test_helpers[!grepl("helper\\_debugging\\.[rR]", torch_test_helpers)]
lapply(torch_test_helpers, source)

rm(mlr_test_helpers)

expect_paramtest = function(paramtest) {
  if (!is.atomic(paramtest)) {
    if (length(paramtest$missing)) {
      info_missing = paste0("- '", paramtest$missing, "'", collapse = "\n")
    } else {
      info_missing = ""
    }
    if (length(paramtest$extra)) {
      info_extra = paste0("- '", paramtest$extra, "'", collapse = "\n")
    } else {
      info_extra = ""
    }
    info = paste0("\nMissing parameters:\n", info_missing, "\nExtra parameters:\n", info_extra)
  }
  expect_true(paramtest, info = info)
}
