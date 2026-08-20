test_that("PipeOpTorchUpsample autotest", {
  po_test = po("nn_upsample", scale_factor = 2)
  task = nano_imagenet()
  graph = po("torch_ingress_ltnsr") %>>% po_test

  expect_pipeop_torch(graph, "nn_upsample", task)
})

test_that("PipeOpTorchUpsample paramtest", {
  res = expect_paramset(po("nn_upsample"), nn_upsample)
  expect_paramtest(res)
})

test_that("PipeOpTorchUpsample works with an explicit size", {
  po_test = po("nn_upsample", size = c(80, 96), mode = "bilinear")
  task = nano_imagenet()
  graph = po("torch_ingress_ltnsr") %>>% po_test

  expect_pipeop_torch(graph, "nn_upsample", task)
})

test_that("shape inference matches the operator", {
  expect_shape_inference("nn_upsample", list(scale_factor = 2), c(2, 3, 8))
  expect_shape_inference("nn_upsample", list(scale_factor = 2), c(2, 3, 8, 8))
  expect_shape_inference("nn_upsample", list(scale_factor = c(2, 3), mode = "bilinear"), c(2, 3, 8, 6))
  expect_shape_inference("nn_upsample", list(size = c(5, 7), mode = "bicubic"), c(2, 3, 8, 8))
  expect_shape_inference("nn_upsample", list(size = 9, mode = "linear"), c(2, 3, 8))
  expect_shape_inference("nn_upsample", list(scale_factor = 2, mode = "trilinear"), c(2, 3, 4, 4, 4))
  # a scale factor that is not a whole number rounds the extent down
  expect_equal(po("nn_upsample", scale_factor = 1.5)$shapes_out(list(c(NA, 3L, 7L)))[[1L]],
    c(NA, 3L, 10L))
})

test_that("an unknown extent stays unknown, but a fixed size does not", {
  # `scale_factor` is relative to the input, so nothing can be said about an unknown extent
  expect_equal(po("nn_upsample", scale_factor = 2)$shapes_out(list(c(NA, 3L, NA)))[[1L]],
    c(NA, 3L, NA))
  # `size` fixes the output extent whatever the input is
  expect_equal(po("nn_upsample", size = 16)$shapes_out(list(c(NA, 3L, NA)))[[1L]], c(NA, 3L, 16L))
})

test_that("shape inference requires exactly one of 'size' and 'scale_factor'", {
  expect_error(po("nn_upsample")$shapes_out(list(c(NA, 3L, 8L))),
    "requires exactly one of 'size' and 'scale_factor'", fixed = TRUE)
  expect_error(po("nn_upsample", size = 4, scale_factor = 2)$shapes_out(list(c(NA, 3L, 8L))),
    "requires exactly one of 'size' and 'scale_factor'", fixed = TRUE)
})

test_that("shape inference checks the rank against the mode", {
  # only "nearest" is defined for more than one number of spatial dimensions
  expect_error(po("nn_upsample", scale_factor = 2, mode = "bilinear")$shapes_out(list(c(NA, 3L, 8L))),
    "requires an input with 4 dimensions", fixed = TRUE)
  expect_error(po("nn_upsample", scale_factor = 2, mode = "linear")$shapes_out(list(c(NA, 3L, 8L, 8L))),
    "requires an input with 3 dimensions", fixed = TRUE)
  expect_error(po("nn_upsample", scale_factor = 2)$shapes_out(list(c(NA, 3L))),
    "requires an input with 3 or 4 or 5 dimensions", fixed = TRUE)
})

test_that("shape inference rejects a parameter of the wrong length or an empty output", {
  expect_error(po("nn_upsample", size = c(4, 5, 6), mode = "bilinear")$shapes_out(list(c(NA, 3L, 8L, 8L))),
    "requires 'size' to have 1 or 2 element(s)", fixed = TRUE)
  expect_error(po("nn_upsample", scale_factor = 0.1)$shapes_out(list(c(NA, 3L, 8L))),
    "which no tensor can have", fixed = TRUE)
})

test_that("shape inference agrees with the module for random shapes and parameters", {
  expect_shape_inference("nn_upsample", params = function() list(scale_factor = sample(2:3, 1L)),
    generators = gen_shape(4L))
  expect_shape_inference("nn_upsample",
    params = function() list(size = sample(4:8, 2L), mode = "bilinear"),
    generators = gen_shape(4L))
  expect_shape_inference("nn_upsample",
    params = function() list(scale_factor = sample(2:3, 1L), mode = "trilinear"),
    generators = gen_shape(5L))
})
