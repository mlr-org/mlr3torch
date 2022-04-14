test_that("TorchOpInput works", {
  task = tsk("iris")
  to = top("input")
  model_args = to$train(list(task))$output
  expect_true(inherits(model_args, "ModelArgs"))
})
