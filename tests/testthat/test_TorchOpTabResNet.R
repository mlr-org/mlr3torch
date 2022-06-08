test_that("TorchOpBlockResNet works", {
  to = top("tab_resnet")
  d_main = 10L
  to$param_set$values = list(
    n_blocks = 2L,
    d_main = d_main,
    d_hidden = 8L,
    dropout_first = 0.2,
    dropout_second = 0.3,
    normalization = "batch_norm",
    activation = "relu",
    skip_connection = TRUE
  )

  inputs = list(
    input = torch_randn(16L, d_main)
  )
  y = torch_randn(16)
  to$build(inputs, task, y)

  block = invoke(nn_block_resnet, .args = to$param_set$values)
  block$forward(inputs$input)
})

