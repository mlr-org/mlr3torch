# {torch} also skipps the tests on CRAN, because downlooading the prebuilt LibTorch binaries
# is now allowed on CRAN. Until {torch} has a workaround for this, we also need to skip
# the tests on CRAN unfortunately.
if (identical(Sys.getenv("TORCH_TEST", unset = "0"), "1")) {
  library("checkmate")
  library("testthat")
  test_check("mlr3torch")
}
