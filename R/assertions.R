assert_torch_optimizer = function(x) {
  assert_r6(x, "TorchOptimizer")
}

assert_descriptor_torch_loss = function(x) {
  assert_r6(x, "DescriptorTorchLoss")
}

assert_descriptor_torch_callback = function(x) {
  assert_r6(x, "DescriptorTorchCallback")
}

assert_descriptor_torch_callbacks = function(x) {
  assert_list(x, types = "DescriptorTorchCallback")
}
