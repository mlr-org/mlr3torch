test_that("layer norm works", {
  n1 = 10L
  n2 = 4L
  task = tsk("iris")
  inputs = list(input = torch_randn(n1, n2))
  y = torch_randn(10L)
  op = top("layer_norm", dims = 1L, elementwise_affine = FALSE)
  c(layer, output) %<-% op$build(inputs, task)
  res = torch_sum(output$output, dim = 2L)
  expect_true(torch_mean(res)$item() <= 0.0001)
})

