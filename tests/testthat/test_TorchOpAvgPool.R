test_that("TorchOpAvgPool1D works", {
  op = top("avg_pool1d", kernel_size = 10L)
  inputs = list(input = torch_randn(16, 10, 10))
  task = tsk("iris")
  op$build(inputs, task)
  expect_torchop(op, inputs, task, "nn_avg_pool1d")
})

test_that("TorchOpAvgPool1D works", {
  op = top("avg_pool2d", kernel_size = 10L)
  inputs = list(input = torch_randn(16, 10, 10, 10))
  task = tsk("iris")
  expect_torchop(op, inputs, task, "nn_avg_pool2d")
})

test_that("TorchOpAvgPool1D works", {
  op = top("avg_pool3d", kernel_size = 10L)
  inputs = list(input = torch_randn(16, 10, 10, 10, 10))
  task = tsk("iris")
  expect_torchop(op, inputs, task, "nn_avg_pool3d")
})
