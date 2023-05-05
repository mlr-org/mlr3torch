test_that("TorchOpFlatten works", {
  op = top("flatten")
  task = tsk("iris")
  param_vals = list(
    start_dim = 2L,
    end_dim = 3L
  )
  inputs = list(input = torch_randn(16, 8, 9))
  op$param_set$values = insert_named(op$param_set$values, param_vals)
  expect_torchop(
    op = op,
    inputs = inputs,
    task = task,
    class = "nn_flatten"
  )
})
