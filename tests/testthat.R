if (identical(Sys.getenv("TORCH_TEST", unset = "0"), "1")) {
  library("checkmate")
  library("testthat")
  library("mlr3torch")
  test_check("mlr3torch")
}
