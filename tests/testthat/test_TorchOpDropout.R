test_that("TorchOpDropout works", {
  op = top("dropout")
  task = tsk("iris")
  for (i in seq_len(3)) {
    param_vals = list(
      p = runif(1),
      inplace = sample(c(TRUE, FALSE), 1)
    )

    ndim = sample(4, 1)
    dims = sample(10, ndim, TRUE)
    inputs = list(input = invoke(torch_randn, .args = dims))

    op$param_set$values = insert_named(op$param_set$values, param_vals)
    expect_torchop(
      op = op,
      inputs = inputs,
      task = task,
      class = "nn_dropout"
    )
  }
})
