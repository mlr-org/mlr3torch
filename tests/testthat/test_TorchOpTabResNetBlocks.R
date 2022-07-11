test_that("TorchOpTabResNetBlocks works", {
  op = top("tab_resnet_blocks")
  task = tsk("iris")

  for (i in seq_len(3)) {
    param_vals = list(
      n_blocks = sample(1:3, 1),
      d_main = sample(1:10, 1),
      d_hidden = sample(1:10, 1),
      dropout_first = runif(1),
      dropout_second = runif(1),
      activation = sample(c("relu", "elu"), 1),
      activation_args = list(inplace = sample(c(TRUE, FALSE), 1)),
      bn.momentum = 0.2,
      skip_connection = sample(c(TRUE, FALSE), 1)
    )
    n_features = sample(1:10, 1)
    n_batch = sample(1:3, 1)
    inputs = list(input = torch_randn(n_batch, n_features))

    expect_torchop(
      op = op,
      inputs = inputs,
      param_vals = param_vals,
      task = task
    )

  }
})
