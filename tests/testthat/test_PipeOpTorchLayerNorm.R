test_that("PipeOpTorch autotest", {
  po_test = po("nn_layer_norm", dims = 1)
  task = tsk("iris")
  graph = po("torch_ingress_num") %>>% po_test
  expect_pipeop_torch(graph, "nn_layer_norm", task, "nn_layer_norm")
})

test_that("PipeOpTorch paramtest", {
  res = expect_paramset(po("nn_layer_norm", dims = 1), nn_layer_norm, exclude = c("normalized_shape", "dims"))
  expect_paramtest(res)
})

test_that("shape inference matches the operator", {
  expect_shapes_out_torch("nn_layer_norm", list(dims = 1), c(2, 7, 16))
  expect_shapes_out_torch("nn_layer_norm", list(dims = 2), c(2, 4, 7, 16))
})

test_that("shape inference requires the normalized dimensions", {
  expect_error(po("nn_layer_norm", dims = 1)$shapes_out(list(c(NA, 7, NA))),
    "requires the last 1 dimension(s), which make up 'normalized_shape', of the input shape",
    fixed = TRUE)
})

test_that("'dims' may go up to the number of input dimensions", {
  # the bound is the number of dimensions of the input shape, not the number of input channels
  obj = po("nn_layer_norm", dims = 3)
  shape_in = list(c(NA, 4L, 7L, 16L))
  expect_equal(obj$shapes_out(shape_in)[[1L]], c(NA, 4L, 7L, 16L))
  module = get_private(obj)$.make_module(shape_in, obj$param_set$get_values(), NULL)
  expect_equal(unlist(module$normalized_shape), c(4L, 7L, 16L))
  expect_equal(dim(module(torch_randn(2, 4, 7, 16))), c(2, 4, 7, 16))
  # `dims` may not exceed the number of dimensions
  expect_error(po("nn_layer_norm", dims = 5)$shapes_out(shape_in), "dims")
})
