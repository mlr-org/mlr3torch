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

  c(layer, output) %<-% top("add")$build(inputs, task, ty)
  expect_true(torch_equal(x1 + x2, output))

  c(layer, output) %<-% top("mul")$build(inputs, task, ty)
  expect_true(torch_equal(x1 * x2, output))

  c(layer, output) %<-% top("cat", dim = 2L)$build(inputs, task, ty)
  expect_true(torch_equal(torch_cat(list(x1, x2), dim = 2L), output))


  c(layer, output) %<-% top("cat", dim = 1L)$build(inputs, task, ty)
  expect_true(torch_equal(torch_cat(list(x1, x2), dim = 1L), output))
})

# TODO: Also test for order here
