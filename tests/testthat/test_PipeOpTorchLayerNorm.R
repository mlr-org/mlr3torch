test_that("PipeOpTorch autotest", {
  po_test = po("nn_layer_norm", dims = 1)
  task = tsk("iris")
  graph = po("torch_ingress_num") %>>% po_test
  autotest_pipeop_torch(graph, "nn_layer_norm", task, "nn_layer_norm")
})

test_that("PipeOpTorch paramtest", {
  res = autotest_paramset(po("nn_layer_norm", dims = 1), nn_layer_norm, exclude = c("normalized_shape", "dims"))
  expect_paramtest(res)
})
