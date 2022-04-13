test_that("TorchOpCustom .build works", {
  fn = function(input, out_features, task) {
    in_features = dim(input[startsWith(names(input), "x")][[1L]])[[2L]]
    nn_linear(in_features = in_features, out_features = out_features)
  }
  top_custom = TorchOpCustom$new()
  top_custom$param_set$values$fn = fn
  top_custom$param_set$values$args = list(out_features = 3)
  top_relu = TorchOpReLU$new()
  architecture = Architecture$new()
  architecture$add_torchop(top_custom)$add_torchop(top_relu)
  tensor = torch_randn(16, 7)
  input = list(x = tensor)
  layer = top_custom$build(input)
  expect_true(class(layer)[[1]] == "nn_linear")
})
