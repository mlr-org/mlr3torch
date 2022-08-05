test_that("TorchOpLayerNorm works", {
  task = tsk("iris")
  op = top("layer_norm")
  inputs = list(input = torch_randn(16, 7, 3))

    param_vals = list(n_dim = 1L, elementwise_affine = TRUE)
  op$param_set$values = insert_named(op$param_set$values, param_vals)

  exclude = c(
    "n_dim", # alternative implementation
    "normalized_shape" # inferred
  )
  expect_torchop(
    op = op,
    inputs = inputs,
    task = task,
    class = "nn_layer_norm",
    exclude = exclude
  )

  param_vals = list(n_dim = 2L, elementwise_affine = FALSE)
  op$param_set$values = insert_named(op$param_set$values, param_vals)
  expect_torchop(
    op = op,
    inputs = inputs,
    task = task,
    class = "nn_layer_norm",
    exclude = exclude
  )
})
