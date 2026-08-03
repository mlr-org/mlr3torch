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

sampler_adaptive_avg_pool = function(dim) {
  list(
    shape_in = sample(20:25, size = dim + 2L, replace = TRUE),
    conv_dim = dim,
    output_size = sample(c(1, dim), size = 1)
  )
}

test_that("adaptive_avg_output_shape works", {
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

test_that("adaptive_avg_output_shape requires a batch dimension", {
  # the PipeOp rejects such a shape via assert_ndim() before the shape function is reached
  expect_error(
    adaptive_avg_output_shape(shape_in = c(3, 20, 20), conv_dim = 2, output_size = 5),
    "length 4"
  )
})

test_that("shape inference matches the operator", {
  expect_shapes_out_torch("nn_adaptive_avg_pool2d", list(output_size = c(2, 3)), c(2, 3, 16, 20))
  expect_shapes_out_torch("nn_adaptive_avg_pool1d", list(output_size = 4), c(2, 3, 17))
})

test_that("an unknown input extent still gives a known output", {
  # the output size is fixed by `output_size`, so rejecting an unknown input extent would throw
  # away information
  for (d in 1:2) {
    obj = po(sprintf("nn_adaptive_avg_pool%id", d), output_size = rep(4L, d))
    shape_in = as.integer(c(NA, 3L, rep(NA_integer_, d)))
    expect_equal(obj$shapes_out(list(input = shape_in))[[1L]], c(NA, 3L, rep(4L, d)))
  }
})
