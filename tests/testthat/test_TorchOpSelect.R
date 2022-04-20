test_that("TorchOpSelect works", {
  task = tsk("boston_housing")
  to = top("select", .types = "cat")
  inputs = list(input = list(num = torch_randn(2, 5), cat = torch_randn(2, 4)))
  y = torch_randn(2, 1)
  c(layer, output) %<-% to$build(inputs, task, y)
  expect_equal(output$shape, c(2, 4L))
})
