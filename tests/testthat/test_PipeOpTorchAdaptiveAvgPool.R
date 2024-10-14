test_that("PipeOpTorchAdaptiveAvgPool1D works", {
  # TODO: pick a good output size, perhaps by matching the output size of non-adaptive average pooling
  po_test = po("nn_adaptive_avg_pool1d", output_size = 10)
  task = tsk("iris")
  graph = po("torch_ingress_num") %>>%
    po("nn_unsqueeze", dim = 2) %>>%
    po_test
  expect_pipeop_torch(graph, "nn_adaptive_avg_pool1d", task)
})

test_that("PipeOpTorchAdaptiveAvgPool1D paramtest", {
  res = expect_paramset(po("nn_adaptive_avg_pool1d"), nn_adaptive_avg_pool1d, exclude = "num_features")
  expect_paramtest(res)
})

test_that("PipeOpTorchAdaptiveAvgPool2D works", {
  # TODO: pick a good output size, perhaps by matching the output size of non-adaptive average pooling
  po_test = po("nn_adaptive_avg_pool2d", output_size = 10)
  task = tsk("iris")
  graph = po("torch_ingress_num") %>>%
    po("nn_unsqueeze", dim = 2) %>>%
    po_test
  expect_pipeop_torch(graph, "nn_adaptive_avg_pool2d", task)
})

test_that("PipeOpTorchAdaptiveAvgPool2D paramtest", {
  res = expect_paramset(po("nn_adaptive_avg_pool2d"), nn_adaptive_avg_pool2d, exclude = "num_features")
  expect_paramtest(res)
})

test_that("PipeOpTorchAdaptiveAvgPool3D works", {
  # TODO: pick a good output size, perhaps by matching the output size of non-adaptive average pooling
  po_test = po("nn_adaptive_avg_pool3d", output_size = 10)
  task = tsk("iris")
  graph = po("torch_ingress_num") %>>%
    po("nn_unsqueeze", dim = 2) %>>%
    po_test
  expect_pipeop_torch(graph, "nn_adaptive_avg_pool3d", task)
})

test_that("PipeOpTorchAdaptiveAvgPool3D paramtest", {
  res = expect_paramset(po("nn_adaptive_avg_pool3d"), nn_adaptive_avg_pool3d, exclude = "num_features")
  expect_paramtest(res)
})