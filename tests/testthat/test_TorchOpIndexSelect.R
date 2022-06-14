test_that("TorchOp works", {
  task = tsk("mtcars")
  y = torch_randn(1)
  x = torch_randn(16, 5, 10)
  op = top("indexselect")
  c(layer, output) %<-% op$build(list(input = x), task)
  expect_equal(output$output$shape, c(16, 1, 10))
})
