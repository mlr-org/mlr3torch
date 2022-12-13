test_that("PipeOpTorchHead autotest", {
  po_test = po("nn_head")
  task = tsk("iris")
  graph = po("torch_ingress_num") %>>% po_test

  autotest_pipeop_torch(graph, "nn_head", task, "nn_linear")
})


test_that("PipeOpTorchHead paramtest", {
  po_test = po("nn_head")
  res = run_paramtest(po_test, torch::nn_linear, exclude = c("out_features", "in_features"))
  expect_paramtest(res)
})
