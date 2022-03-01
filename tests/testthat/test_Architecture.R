test_that("Can build NeuralNetwork from Architecture", {
  # DataBackendTorchDataTable$debug("dataloader")
  task = tsk("mtcars")
  architecture = Architecture$new()
  architecture$add("linear", list(out_features = 10))
  architecture$add("relu")
  output = reduce_architecture(architecture, task)
  network = output[["network"]]
})
