test_that("PipeOpTorchBatchNorm1D autotest", {
  po_test = po("nn_batch_norm1d")
  task = tsk("iris")
  graph1 = po("torch_ingress_num") %>>% po_test
  graph2 = po("torch_ingress_num") %>>% po("nn_unsqueeze", dim = 2) %>>% po_test

  autotest_pipeop_torch(graph1, "nn_batch_norm1d", task)
  autotest_pipeop_torch(graph2, "nn_batch_norm1d", task)
})

test_that("PipeOpTorchBatchNorm1D paramtest", {
  res = run_paramtest(po("nn_batch_norm1d"), nn_batch_norm1d, exclude = "num_features")
  expect_paramtest(res)
})

test_that("PipeOpTorchBatchNorm2D autotest", {
  po_test = po("nn_batch_norm2d")
  task = tsk("nano_imagenet")
  graph = po("torch_ingress_img", channels = 3, width = 64, height = 64) %>>% po_test

  autotest_pipeop_torch(graph, "nn_batch_norm2d", task)
})

test_that("PipeOpTorchBatchNorm2D paramtest", {
  res = run_paramtest(po("nn_batch_norm2d"), nn_batch_norm2d, exclude = "num_features")
  expect_paramtest(res)
})

test_that("PipeOpTorchBatchNorm3D autotest", {
  po_test = po("nn_batch_norm3d")
  task = tsk("nano_imagenet")
  graph = po("torch_ingress_img", channels = 3, width = 64, height = 64) %>>%
    po("nn_reshape", shape = c(NA, 3, 64, 8, 8)) %>>%
    po_test

  autotest_pipeop_torch(graph, "nn_batch_norm3d", task)
})

test_that("PipeOpTorchBatchNorm3D paramtest", {
  res = run_paramtest(po("nn_batch_norm3d"), nn_batch_norm3d, exclude = "num_features")
  expect_paramtest(res)
})
