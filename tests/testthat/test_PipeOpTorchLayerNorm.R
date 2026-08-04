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

test_that("'dims' may not reach the batch dimension", {
  # normalizing over the last 3 of 3 dimensions would include the batch dimension, which is not a
  # feature dimension; `dims` below 1 is already rejected by the parameter set
  expect_error(po("nn_layer_norm", dims = 3)$shapes_out(list(c(NA, 7L, 16L))),
    "would include the batch dimension", fixed = TRUE)
  expect_error(po("nn_layer_norm", dims = 5)$shapes_out(list(c(NA, 7L, 16L))),
    "would include the batch dimension", fixed = TRUE)
  expect_error(po("nn_layer_norm", dims = 0), "not >= 0.5", fixed = TRUE)
  # the largest permitted value normalizes over everything but the batch dimension
  expect_equal(po("nn_layer_norm", dims = 2)$shapes_out(list(c(NA, 7L, 16L)))[[1L]], c(NA, 7L, 16L))
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
