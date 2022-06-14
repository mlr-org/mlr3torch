test_that("TorchOpBlockResNet works", {
  to = top("tab_resnet_block")
  d_main = 10L
  to$param_set$values = list(
    n_blocks = 2L,
    d_main = d_main,
    d_hidden = 8L,
    dropout_first = 0.2,
    dropout_second = 0.3,
    activation = "relu"
  )

  inputs = list(
    input = torch_randn(16L, d_main)
  )
  y = torch_randn(16)
  out = to$build(inputs, task)
  layer = out$layer

  expect_error(layer(inputs$input), regexp = NA)
})
