test_that("TorchOpMerge works", {
  task = tsk("iris")
  to_add = top("add")
  to_mul = top("mul")
  to_cat1 = top("cat", dim = 1L)
  to_cat2 = top("cat", dim = 1L)
  x1 = torch_randn(1, 3)
  x2 = torch_randn(1, 3)
  y = torch_randn(1)
  inputs = list(input1 = x1, input2 = x2)
  c(layer, output) %<-% to_add$build(inputs, task, y)
  c(layer, output) %<-% to_mul$build(inputs, task, y)
  c(layer, output) %<-% to_mul$build(inputs, task, y)
  expect_true(torch_equal(x1 + x2, output))
})

# TODO: Also test for order here
