test_that("make_paramset works for classif", {
  for (optimizer in torch_reflections$optimizer) {
    expect_error(make_paramset("classif", optimizer, "cross_entropy"), regexp = NA)
  }
})

test_that("make_paramset works for regr", {
  for (optimizer in torch_reflections$optimizer) {
    expect_error(make_paramset("regr", optimizer, "mse"), regexp = NA)
  }
})

# TODO: Add all loss functions once they are implementd
