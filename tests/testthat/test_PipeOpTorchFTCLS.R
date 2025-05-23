test_that("PipeOpTorchFTCLS autotest", {
  task = tsk("iris")
  graph = po("torch_ingress_num") %>>%
   po("nn_tokenizer_num", d_token = 10) %>>%
   po("nn_ft_cls", initialization = "uniform")

  expect_pipeop_torch(graph, "nn_ft_cls", task)
})

test_that("PipeOpTorchFTCLS works for tensors of specified dimensions", {
  # the canonical case: tensor of shape c(batch_size, n_features, d_token)
  task = tsk("iris")
  batch_size = 3
  d_token = 10
  tnsr = torch_tensor(as.matrix(task$data()[seq_len(batch_size), .(Petal.Width, Petal.Length, Sepal.Width, Sepal.Length)]))

  graph = po("torch_ingress_num") %>>%
    po("nn_tokenizer_num", d_token = d_token) %>>%
    po("nn_ft_cls", initialization = "uniform")
  md = graph$train(task)[[1L]]
  net = nn_graph(md$graph, shapes_in = list(torch_ingress_num.input = c(NA, task$n_features, d_token)))

  tnsr_out = net(tnsr)

  # the resulting tensor has an extra feature
  expect_equal(tnsr_out$shape, c(batch_size, task$n_features + 1, d_token))
})