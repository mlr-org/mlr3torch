assert_torch_optimizer = function(x) {
  assert_r6(x, "TorchOptimizer")
}

assert_torch_loss = function(x) {
  assert_r6(x, "TorchLoss")
}

assert_torch_callback = function(x) {
  assert_r6(x, "TorchCallback")
}

assert_torch_callbacks = function(x) {
  assert_list(x, types = "TorchCallback")
}
