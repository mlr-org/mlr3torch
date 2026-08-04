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
  expect_shape_inference("nn_layer_norm", list(dims = 1), c(2, 7, 16))
  expect_shape_inference("nn_layer_norm", list(dims = 2), c(2, 4, 7, 16))
})

test_that("shape inference requires the normalized dimensions", {
  expect_error(po("nn_layer_norm", dims = 1)$shapes_out(list(c(NA, 7, NA))),
    "requires the last 1 dimension(s), which make up 'normalized_shape', of the input shape",
    fixed = TRUE)
})

test_that("shape inference agrees with the module for random shapes and parameters", {
  expect_shape_inference("nn_layer_norm", params = function() list(dims = sample(1:3, 1L)),
    generators = gen_shape(4L))
})
