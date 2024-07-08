library(testthat)
library(mlr3torch)

if (identical(Sys.getenv("TORCH_TEST", unset = "0"), "1")) {
  test_check("torch")
}
