test_that("TorchOpFlatten works", {
  task = tsk("boston_housing")
  inputs = list(input = torch_randn(10, 3, 4))
  to = top("flatten")
  to$build(inputs, task, NULL)
  output = layer(tensor)
  expect_equal(output$shape, c(10L, 12L))
})
