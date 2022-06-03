test_that("TorchOpOutput works", {
  inputs = list(input = torch_randn(10, 5))
  y = torch_randn(10)

  task = tsk("mtcars")
  to = top("output")
  res = to$build(inputs = inputs, task = task, y = y)
  layer = res$layer
  output = res$output
  expect_true(all(dim(layer$parameters$weight) == c(1, 5)))

  task = tsk("iris")

  res = to$build(inputs = inputs, task = task, y = y)
  layer = res$layer
  output = res$output
  expect_true(all(dim(layer$parameters$weight) == c(3, 5)))


  inputs = list(input = torch_randn(10, 5, 7))

  expect_error(to$build(inputs = inputs, task = task, y = y))
})
