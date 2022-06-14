test_that("TorchOpBatchNorm works", {
  bnop = top("batch_norm")
  x = torch_abs(torch_randn(10, 3))
  c(layer, output) %<-% bnop$build(list(input = x))
  y = output$output
  expect_true(all(colMeans(as.array(x)) >= colMeans(as.array(y))))

  bnop = top("batch_norm")
  x = torch_abs(torch_randn(10, 3))
  c(layer, output) %<-% bnop$build(list(input = x))
  y = layer$forward(x)
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


  layer$eval()
  layer$forward(x)
})

test_that("running_var is not -nan", {
  bnop = top("batch_norm")
  x = torch_abs(torch_randn(10, 3))
  c(layer, output) %<-% bnop$build(list(input = x))
  y = output$output
  expect_true(all(colMeans(as.array(x)) >= colMeans(as.array(y))))

  bn = nn_batch_norm1d(10)
  x = torch_randn(16, 10)
  lin = nn_linear(10, 1)

  opt = optim_adam(bn$parameters)

  y_true = torch_randn(16, 1)
  y_hat = lin(bn(x))
  loss_fn = nn_mse_loss()
  loss = loss_fn(y_true, y_hat)
  opt$step()

})
