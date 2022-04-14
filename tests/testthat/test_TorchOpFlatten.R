test_that("TorchOpFlatten works", {
  task = tsk("boston_housing")
  tensor = torch_randn(10, 3, 4)
  layer = top("flatten")$build(tensor, task, NULL)
  output = layer(tensor)
  expect_equal(output$shape, c(10L, 12L))
})
