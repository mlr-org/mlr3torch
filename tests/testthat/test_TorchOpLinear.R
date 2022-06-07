test_that("TorchOpLinear works in 2D", {
  d = 10
  to = top("linear", out_features = d)
  x = torch_randn(1, 5)
  y = torch_randn(1)
  task = tsk("iris")
  c(layer, output) %<-% to$build(list(input = x), task, y)
  expect_equal(output$output$shape, c(1, 10))
})

test_that("TorchOpLinear works in 3D", {
  d = 10
  to = top("linear", out_features = d)
  x = torch_randn(1, 5, 7)
  y = torch_randn(1)
  task = tsk("iris")
  c(layer, output) %<-% to$build(list(input = x), task, y)
  expect_equal(output$output$shape, c(1, 5, 10))
})
