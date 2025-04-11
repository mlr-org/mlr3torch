test_that("PipeOpTorchFTCLS autotest", {
  po_cls = po("nn_ft_cls", d_token = 10, initialization = "uniform")
  task = tsk("iris")
  graph = po("torch_ingress_num") %>>% po_cls

  expect_pipeop_torch(graph, "nn_ft_cls", task)
})