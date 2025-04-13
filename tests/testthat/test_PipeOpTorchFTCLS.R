test_that("PipeOpTorchFTCLS autotest", {
  # TODO: determine whether the autotest will require a specific d_token
  po_cls = po("nn_ft_cls", d_token = 10, initialization = "uniform")
  task = tsk("iris")
  graph = po("torch_ingress_num") %>>% po_cls
  # autotest appears to be failing on an input with shape c(1, 4)
  # attempts to concatenate a tensor with shape c(1, 1, d_token) to this input tensor
  # and this fails because they need to have the same number of dimensions
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

# LLM test
test_that("PipeOpTorchFTCLS works for tensors of specified dimensions", {
  # Setup the CLS token PipeOp with a small token dimension
  d_token <- 4
  po_cls <- po("nn_ft_cls", d_token = d_token, initialization = "uniform")
  
  # Case 1: The canonical case - input tensor of shape c(batch_size, n_features, d_token)
  batch_size <- 5
  n_features <- 10
  input_3d <- torch_randn(c(batch_size, n_features, d_token))
  
  # The output should have shape c(batch_size, n_features + 1, d_token)
  output_3d <- po_cls$train(list(input_3d))[[1]]
  expect_equal(output_3d$shape, c(batch_size, n_features + 1, d_token))
  
  # Ensure the original features are preserved
  expect_true(torch_allclose(output_3d[, 1:n_features, ], input_3d))
  
  # Case 2: 2D input tensor of shape c(batch_size, n_features)
  # This is the case that was failing in autotest
  input_2d <- torch_randn(c(batch_size, n_features))
  
  # Should reshape to add the CLS token
  output_2d <- po_cls$train(list(input_2d))[[1]]
  
  # Check dimensions - note the output will likely be expanded to 3D
  expect_equal(output_2d$shape[1], batch_size)
  expect_equal(output_2d$shape[2], n_features + 1)
  
  # Case 3: 4D input tensor (e.g., image data) c(batch_size, channels, height, width)
  channels <- 3
  height <- 8
  width <- 8
  input_4d <- torch_randn(c(batch_size, channels, height, width))
  
  # Check if it can handle 4D input
  expect_error(output_4d <- po_cls$train(list(input_4d))[[1]], NA) # Expect no error
  
  # The CLS token should be added along the channel dimension
  if (exists("output_4d")) {
    expect_equal(output_4d$shape[1], batch_size)
    expect_equal(output_4d$shape[2], channels + 1)
    expect_equal(output_4d$shape[3], height)
    expect_equal(output_4d$shape[4], width)
  }
  
  # Case 4: Single sample (batch_size = 1)
  # This was specifically mentioned as failing
  input_single <- torch_randn(c(1, n_features, d_token))
  output_single <- po_cls$train(list(input_single))[[1]]
  expect_equal(output_single$shape, c(1, n_features + 1, d_token))
  
  # Case 5: Very small features (edge case)
  input_small <- torch_randn(c(batch_size, 1, d_token))
  output_small <- po_cls$train(list(input_small))[[1]]
  expect_equal(output_small$shape, c(batch_size, 2, d_token))
})