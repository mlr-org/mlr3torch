test_that("TorchOpFlatten works", {
  task = tsk("boston_housing")
  inputs = list(input = torch_randn(10, 3, 4))
  to = top("flatten")
  layer = to$build(inputs, task)$layer
  output = layer(torch_randn(10, 3, 4))
  expect_equal(output$shape, c(10L, 12L))
})
