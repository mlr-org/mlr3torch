test_that("All activation functions can be constructed", {
  for (act in torch_reflections$activation) {
    expect_r6(top(sprintf("%s_1", act)), "TorchOp")
  }
})

test_that("TorchOpActivation works", {
  act = top("activation", fn = "relu", args = list(inplace = TRUE))
  f = get_private(act)$.build(NULL, act$param_set$values)

  # this tests that the inplace parameters is properly passed
  x = torch_randn(10, 3)
  y = f(x)
  expect_true(torch_equal(x, y))
})
