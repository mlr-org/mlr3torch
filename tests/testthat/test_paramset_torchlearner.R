test_that("paramset_torchlearner has parameters properly documented", {
  param_set = paramset_torchlearner()

  template = readLines(system.file("./man-roxygen/paramset_torchlearner.R", package = "mlr3torch"))

  pardoc = regmatches(template, regexpr("(?<=\\`)[a-zA-Z][a-zA-Z0-9_]+(?=\\` ::)", template, perl = TRUE))

  expect_true(length(pardoc) == param_set$length)
  expect_set_equal(pardoc, param_set$ids())
})
