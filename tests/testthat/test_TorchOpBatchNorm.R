test_that("TorchOpBatchNorm works", {
  bnop = top("batch_norm")
  x = torch_abs(torch_randn(10, 3))
  c(layer, output) %<-% bnop$build(list(input = x))
  y = output$output
  expect_true(all(colMeans(as.array(x)) >= colMeans(as.array(y))))

  bnop = top("batch_norm")
  x = torch_abs(torch_randn(10, 3))
  c(layer, output) %<-% bnop$build(list(input = x))
  y = output$output
  s = torch_sum(y)
  expect_true(s$item() <= 0.0001)
  expect_true(inherits(layer, "nn_batch_norm1d"))

  x = torch_abs(torch_randn(10, 3, 4))
  c(layer, output) %<-% bnop$build(list(input = x))
  s = torch_sum(y)
  expect_true(s$item() <= 0.0001)
  expect_true(inherits(layer, "nn_batch_norm1d"))

  x = torch_abs(torch_randn(10, 3, 4, 5))
  c(layer, output) %<-% bnop$build(list(input = x))
  s = torch_sum(y)
  expect_true(s$item() <= 0.0001)
  expect_true(inherits(layer, "nn_batch_norm2d"))

  x = torch_abs(torch_randn(10, 3, 4, 5, 6))
  c(layer, output) %<-% bnop$build(list(input = x))
  s = torch_sum(y)
  expect_true(s$item() <= 0.0001)
  expect_true(inherits(layer, "nn_batch_norm3d"))
})
