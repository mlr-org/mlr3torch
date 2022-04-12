test_that("make_paramset works for classif", {
  for (optimizer in torch_reflections$optimizer) {
    expect_error(make_paramset(optimizer, "classif"), regexp = NA)
  }
})

test_that("make_paramset works for regr", {
  for (optimizer in torch_reflections$optimizer) {
    expect_error(make_paramset(optimizer, "regr"), regexp = NA)
  }
})
