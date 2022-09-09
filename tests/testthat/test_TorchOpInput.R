test_that("TorchOpInput works", {
  task = tsk("iris")
  to = top("input")
  model_args = to$train(list(task))$output
  expect_true(inherits(model_args, "ModelConfig"))

  g = top("input") %>>% top("tab_tokenizer", d_token = 3L)
  out = g$train(task)
})
