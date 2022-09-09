test_that("TorchOpMaxPool1D works", {
  task = tsk("iris")
  op = top("max_pool1d", kernel_size = 5L)
  inputs = list(input = torch_randn(16, 10, 10))
  expect_torchop(op, inputs, tsk("iris"), "nn_max_pool1d")
})

test_that("TorchOpMaxPool2D works", {
  task = tsk("iris")
  op = top("max_pool2d", kernel_size = c(1L, 2L))
  inputs = list(input = torch_randn(16, 10, 10, 8))
  expect_torchop(op, inputs, tsk("iris"), "nn_max_pool2d")
})

test_that("TorchOpMaxPool3D works", {
  task = tsk("iris")
  op = top("max_pool3d", kernel_size = c(1L, 2L, 3L))
  inputs = list(input = torch_randn(16, 10, 10, 8, 7))
  expect_torchop(op, inputs, tsk("iris"), "nn_max_pool3d")
})
