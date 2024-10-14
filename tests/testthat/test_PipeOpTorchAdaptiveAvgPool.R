test_that("PipeOpTorchAdaptiveAvgPool1D works", {
  po_test = po("nn_adaptive_avg_pool1d", output_size = 10)
  task = tsk("iris")
  graph = po("torch_ingress_num") %>>%
    po("nn_unsqueeze", dim = 2) %>>%
    po_test
  expect_pipeop_torch(graph, "nn_adaptive_avg_pool1d", task)
})

test_that("PipeOpTorchAdaptiveAvgPool1D paramtest", {
  res = expect_paramset(po("nn_adaptive_avg_pool1d"), nn_adaptive_avg_pool1d, exclude = "num_features")
  expect_paramtest(res)
})

test_that("PipeOpTorchAdaptiveAvgPool2D works with a 1d output size", {
  po_test = po("nn_adaptive_avg_pool2d", output_size = 10)
  task = nano_imagenet()
  graph = po("torch_ingress_ltnsr") %>>% po_test

  expect_pipeop_torch(graph, "nn_adaptive_avg_pool2d", task)
})

test_that("PipeOpTorchAdaptiveAvgPool2D works with a 2d output size", {
  po_test = po("nn_adaptive_avg_pool2d", output_size = c(8, 12))
  task = nano_imagenet()
  graph = po("torch_ingress_ltnsr") %>>% po_test

  expect_pipeop_torch(graph, "nn_adaptive_avg_pool2d", task)
})

test_that("PipeOpTorchAdaptiveAvgPool2D paramtest", {
  res = expect_paramset(po("nn_adaptive_avg_pool2d"), nn_adaptive_avg_pool2d, exclude = "num_features")
  expect_paramtest(res)
})

test_that("PipeOpTorchAdaptiveAvgPool3D works with a 1d output size", {
  po_test = po("nn_adaptive_avg_pool3d", output_size = 10)
  task = nano_imagenet()
  graph = po("torch_ingress_ltnsr") %>>%
    po("nn_reshape", shape = c(NA, 3, 64, 8, 8)) %>>%
    po_test

  expect_pipeop_torch(graph, "nn_adaptive_avg_pool3d", task)
})

test_that("PipeOpTorchAdaptiveAvgPool3D works with a 3d output size", {
  po_test = po("nn_adaptive_avg_pool3d", output_size = c(10, 11, 12))
  task = nano_imagenet()
  graph = po("torch_ingress_ltnsr") %>>%
    po("nn_reshape", shape = c(NA, 3, 64, 8, 8)) %>>%
    po_test

  expect_pipeop_torch(graph, "nn_adaptive_avg_pool3d", task)
})

test_that("PipeOpTorchAdaptiveAvgPool3D paramtest", {
  res = expect_paramset(po("nn_adaptive_avg_pool3d"), nn_adaptive_avg_pool3d, exclude = "num_features")
  expect_paramtest(res)
})

sampler_adaptive_avg_pool = function(dim, batch = TRUE) {
  list(
    shape_in = sample(20:25, size = dim + 1 + as.integer(batch), replace = TRUE),
    conv_dim = dim,
    output_size = sample(c(1, dim), size = 1)
  )
}

test_that("adaptive_avg_output_shape works when there is a batch dimension", {
  for (dim in 1:3) {
    testcase = sampler_adaptive_avg_pool(dim)
    mg = switch(dim,
      nn_adaptive_avg_pool1d,
      nn_adaptive_avg_pool2d,
      nn_adaptive_avg_pool3d
    )

    m = do.call(mg, testcase[names(testcase) %in% formalArgs(mg)])
    outshape = with_no_grad(m(do.call(torch::torch_randn, args = list(unname(testcase$shape_in)))))$shape
    expect_true(all(outshape == do.call(adaptive_avg_output_shape, args = testcase)))
  }
})

test_that("adaptive_avg_output_shape works when there is no batch dimension", {
  for (dim in 1:3) {
    testcase = sampler_adaptive_avg_pool(dim, batch = FALSE)
    mg = switch(dim,
      nn_adaptive_avg_pool1d,
      nn_adaptive_avg_pool2d,
      nn_adaptive_avg_pool3d
    )
    m = do.call(mg, testcase[names(testcase) %in% formalArgs(mg)])
    outshape = with_no_grad(m(do.call(torch::torch_randn, args = list(unname(testcase$shape_in)))))$shape
    expect_warning(shape <<- do.call(adaptive_avg_output_shape, args = testcase), regexp = "batch dimension")
    expect_true(all(outshape == shape))
  }
})
