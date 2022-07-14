test_that("TorchOpSelect works", {
  task = tsk("iris")
  y = torch_randn(10)
  inputs = list(
    input = list(img = torch_randn(10, 8, 8), num = torch_randn(10, 5), cat = torch_randn(10, 7))
  )
  to = top("select", items = "img")
  res = to$build(inputs, task)
  out = res$output$output
  layer = res$layer

  x = layer(list(img = 1))
  expect_true(x == 1)
  expect_true(all(out$shape == c(10, 8, 8)))
})
