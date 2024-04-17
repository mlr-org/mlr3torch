test_that("PipeOpTorchLinear works", {
  po_linear = po("nn_linear", out_features = 10)
  graph = po("torch_ingress_num") %>>% po_linear
  task = tsk("iris")

  expect_pipeop_torch(graph, "nn_linear", task, "nn_linear")
})

test_that("PipeOpTorchLinear paramtest", {
  po_linear = po("nn_linear", out_features = 10)
  res = expect_paramset(po_linear, nn_linear, exclude = "in_features")
  expect_paramtest(res)
})
