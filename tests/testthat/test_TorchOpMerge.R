test_that("TorchOpMerge works", {
  task = tsk("iris")
  to = top("merge", .method = "add")
  x1 = torch_randn(1, 3)
  x2 = torch_randn(1, 3)
  y = torch_randn(1)
  inputs = list(input1 = x1, input2 = x2)
  c(layer, output) %<-% to$build(inputs, task, y)
  expect_true(torch_equal(x1 + x2, output))
})

# TODO: Also test for order here
