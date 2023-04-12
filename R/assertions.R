

assert_torch_optimizer = function(x) {
  expect_r6(x, "TorchOptimizer")
}

assert_torch_loss = function(x) {
  expect_r6(x, "TorchLoss")
}

assert_torch_callback = function(x) {
  expect_r6(x, "TorchCallback")
}

assert_torch_callbacks = function(x) {
  expect_list(x, types = "TorchCallback")
}
