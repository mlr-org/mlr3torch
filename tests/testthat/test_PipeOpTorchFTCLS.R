test_that("PipeOpTorchFTCLS autotest", {
  po_cls = po("nn_ft_cls", d_token = 10, initialization = "uniform")
  task = tsk("iris")
  graph = po("torch_ingress_num") %>>% po_cls

  expect_pipeop_torch(graph, "nn_ft_cls", task)
})

test_that("PipeOpTorchFTCLS works for tensors of specified dimensions", {
  # autotest appears to be failing on an input with shape c(1, 4)

  # the canonical case: tensor of shape c(NA, n_features, d_token)

  # 
})