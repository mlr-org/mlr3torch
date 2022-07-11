test_that("TorchOpLayerNorm works", {
  task = tsk("iris")
  op = top("layer_norm")
  inputs = list(input = torch_randn(16, 7, 3))

  expect_torchop(
    op = op,
    inputs = inputs,
    param_vals = list(n_dim = 1L, elementwise_affine = TRUE),
    task = task
  )

  expect_torchop(
    op = op,
    inputs = inputs,
    param_vals = list(n_dim = 2L, elementwise_affine = FALSE),
    task = task
  )
})
