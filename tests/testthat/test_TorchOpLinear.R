test_that("TorchOpLinear works", {
  task = tsk("iris")
  op = top("linear")

  for (i in seq_len(3)) {
    bias = sample(c(TRUE, FALSE), 1)
    out_features = sample(1:20, 1)
    ndim = sample(2:5, 1)
    dims = sample.int(20, ndim)
    inputs = list(input = invoke(torch_randn, .args = dims))

    param_vals = list(out_features = out_features)

    op$param_set$values = insert_named(op$param_set$values, param_vals)
    expect_torchop(
      op = op,
      inputs = inputs,
      task = task,
      class = "nn_linear",
      exclude = "in_features"
    )
  }
})
