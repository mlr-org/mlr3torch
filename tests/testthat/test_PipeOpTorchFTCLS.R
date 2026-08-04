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

test_that("shape inference matches the operator", {
  expect_shape_inference("nn_ft_cls", list(initialization = "uniform"), c(2, 7, 16))
})

test_that("shape inference requires the token dimension", {
  expect_error(po("nn_ft_cls")$shapes_out(list(input = c(NA, 7, NA))),
    "requires the token dimension (dimension 3) of the input shape to be known", fixed = TRUE)
})

test_that("shape inference agrees with the module for random shapes and parameters", {
  expect_shape_inference("nn_ft_cls", list(initialization = "uniform"), generators = gen_shape(3L))
})
