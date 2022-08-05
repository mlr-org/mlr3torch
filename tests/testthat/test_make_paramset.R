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

test_that("make_paramset_module works", {
  net = nn_module(
    initialize = function(d_in, d_out) {
      self$linear = nn_linear(d_in, d_out)
    },
    forward = function(x) {
      self$linear(x)
    }
  )
  x = make_paramset_module(net)
  expect_true(all(x$ids() == c("net.d_in", "net.d_out")))
  expect_true(x$tags$net.d_in == "network")
  expect_true(x$tags$net.d_out == "network")

})
