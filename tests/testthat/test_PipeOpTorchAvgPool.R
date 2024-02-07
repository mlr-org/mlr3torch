test_that("PipeOpTorchAvgPool1D works", {
  po_test = po("nn_avg_pool1d", kernel_size = 2)
  task = tsk("iris")
  graph = po("torch_ingress_num") %>>%
    po("nn_unsqueeze", dim = 2) %>>%
    po_test
  expect_pipeop_torch(graph, "nn_avg_pool1d", task)
})

test_that("PipeOpTorchAvgPool1D paramtest", {
  res = expect_paramset(po("nn_avg_pool1d"), nn_avg_pool1d, exclude = "num_features")
  expect_paramtest(res)
})

test_that("PipeOpTorchAvgPool2D autotest", {
  po_test = po("nn_avg_pool2d", kernel_size = 3)
  task = nano_imagenet()
  graph = po("torch_ingress_ltnsr") %>>% po_test

  expect_pipeop_torch(graph, "nn_avg_pool2d", task)
})

test_that("PipeOpTorchAvgPool2D paramtest", {
  res = expect_paramset(po("nn_avg_pool2d"), nn_avg_pool2d, exclude = "num_features")
  expect_paramtest(res)
})

test_that("PipeOpTorchAvgPool3D autotest", {
  po_test = po("nn_avg_pool3d", kernel_size = c(2, 3, 4))
  task = nano_imagenet()
  graph = po("torch_ingress_ltnsr") %>>%
    po("nn_reshape", shape = c(NA, 3, 64, 8, 8)) %>>%
    po_test

  expect_pipeop_torch(graph, "nn_avg_pool3d", task)
})

test_that("PipeOpTorchAvgPool3D paramtest", {
  res = expect_paramset(po("nn_avg_pool3d"), nn_avg_pool3d, exclude = "num_features")
  expect_paramtest(res)
})

sampler_avg_pool = function(dim, batch = TRUE) {
  list(
    shape_in = sample(20:25, size = dim + 1 + as.integer(batch), replace = TRUE),
    conv_dim = dim,
    padding = sample(1:2, size = dim, replace = TRUE),
    stride = sample(1:3, size = dim, replace = TRUE),
    kernel_size = sample(5:6, size = dim, replace = TRUE),
    ceil_mode = sample(c(TRUE, FALSE), 1)
  )
}

test_that("avg_output_shape works when there is a batch dimension", {
  for (dim in 1:3) {
    testcase = sampler_avg_pool(dim)
    mg = switch(dim,
      nn_avg_pool1d,
      nn_avg_pool2d,
      nn_avg_pool3d
    )
    m = do.call(mg, testcase[names(testcase) %in% formalArgs(mg)])
    outshape = with_no_grad(m(do.call(torch::torch_randn, args = list(unname(testcase$shape_in)))))$shape
    expect_true(all(outshape == do.call(avg_output_shape, args = testcase)))
  }
})

test_that("avg_output_shape works when there is no batch dimension", {
  for (dim in 1:3) {
    testcase = sampler_avg_pool(dim, batch = FALSE)
    mg = switch(dim,
      nn_avg_pool1d,
      nn_avg_pool2d,
      nn_avg_pool3d
    )
    m = do.call(mg, testcase[names(testcase) %in% formalArgs(mg)])
    outshape = with_no_grad(m(do.call(torch::torch_randn, args = list(unname(testcase$shape_in)))))$shape
    expect_warning(shape <<- do.call(avg_output_shape, args = testcase), regexp = "batch dimension")
    expect_true(all(outshape == shape))
  }
})
