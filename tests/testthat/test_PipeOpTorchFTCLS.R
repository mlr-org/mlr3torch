test_that("PipeOpTorchFTCLS autotest", {
  # TODO: determine whether the autotest will require a specific d_token
  po_cls = po("nn_ft_cls", d_token = 10, initialization = "uniform")
  task = tsk("iris")
  graph = po("torch_ingress_num") %>>% po_cls
  # autotest appears to be failing on an input with shape c(1, 4)
  # the PipeOp attempts to concatenate a tensor with shape c(1, 1, d_token) to this input tensor
  # and this fails because they need to have the same number of dimensions
  # TODO: add the feature tokenizer to the graph for this test?
  expect_pipeop_torch(graph, "nn_ft_cls", task)
})

test_that("PipeOpTorchFTCLS works for tensors of specified dimensions", {
  # the canonical case: tensor of shape c(batch_size, n_features, d_token)
  task = tsk("iris")
  batch_size = 3
  d_token = 10
  tnsr = torch_randn(c(batch_size, task$n_features, d_token))

  graph = po("torch_ingress_num") %>>% po("nn_ft_cls", d_token = 10, initialization = "uniform")
  md = graph$train(task)[[1L]]
  net = nn_graph(md$graph, shapes_in = list(torch_ingress_num.input = c(NA, task$n_features)))

  tnsr_out = net(tnsr)

  # the resulting tensor has an extra feature
  expect_equal(tnsr_out$shape, c(batch_size, task$n_features + 1, d_token))
})
