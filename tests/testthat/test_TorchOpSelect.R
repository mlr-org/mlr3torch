test_that("TorchOpSelect works", {
  task = tsk("iris")
  y = torch_randn(10)
  inputs = list(
    input = list(img = torch_randn(10, 8, 8), num = torch_randn(10, 5), cat = torch_randn(10, 7))
  )
  to = top("select", .items = "img")
  out = to$build(inputs, task)$output$output
  expect_true(all(out$shape == c(10, 8, 8)))

})
