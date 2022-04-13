test_that("make_paramset works for classif", {
  for (optimizer in torch_reflections$optimizer) {
    expect_error(make_standard_paramset("classif", optimizer), regexp = NA)
  }
})

test_that("make_paramset works for regr", {
  for (optimizer in torch_reflections$optimizer) {
    expect_error(make_standard_paramset("regr", optimizer), regexp = NA)
  }
})
