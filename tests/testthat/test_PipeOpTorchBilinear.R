test_that("PipeOpTorchBilinear autotest", {
  po_test = po("nn_bilinear", out_features = 7)
  task = tsk("iris")
  graph = po("torch_ingress_num") %>>%
    list(po("nn_linear_1", out_features = 10), po("nn_linear_2", out_features = 12)) %>>%
    po_test

  expect_pipeop_torch(graph, "nn_bilinear", task)
})

test_that("PipeOpTorchBilinear paramtest", {
  # in1_features and in2_features are inferred from the input shapes
  res = expect_paramset(po("nn_bilinear"), nn_bilinear, exclude = c("in1_features", "in2_features"))
  expect_paramtest(res)
})

test_that("PipeOpTorchBilinear has two input channels", {
  obj = po("nn_bilinear", out_features = 7)
  expect_equal(obj$input$name, c("input1", "input2"))
  expect_equal(obj$output$name, "output")
})

test_that("shape inference matches the operator", {
  expect_shape_inference("nn_bilinear", list(out_features = 7), c(2, 4), n_in = 2L)
  expect_shape_inference("nn_bilinear", list(out_features = 7), c(2, 5, 4), n_in = 2L)
  expect_shape_inference("nn_bilinear", list(out_features = 3, bias = FALSE), c(2, 4), n_in = 2L)
})

test_that("shape inference needs both feature dimensions", {
  shapes_out = function(s1, s2) {
    po("nn_bilinear", out_features = 7)$shapes_out(list(input1 = s1, input2 = s2))[[1L]]
  }
  expect_equal(shapes_out(c(NA, 4L), c(NA, 6L)), c(NA, 7L))
  # the last dimensions become in1_features and in2_features, which the module needs to size itself
  expect_error(shapes_out(c(NA, NA), c(NA, 6L)), "'in1_features'", fixed = TRUE)
  expect_error(shapes_out(c(NA, 4L), c(NA, NA)), "'in2_features'", fixed = TRUE)
})

test_that("shape inference requires the leading dimensions to agree", {
  shapes_out = function(s1, s2) {
    po("nn_bilinear", out_features = 7)$shapes_out(list(input1 = s1, input2 = s2))[[1L]]
  }
  expect_error(shapes_out(c(NA, 5L, 4L), c(NA, 6L, 4L)),
    "differ in dimension 2", fixed = TRUE)
  expect_error(shapes_out(c(NA, 4L), c(NA, 5L, 6L)),
    "same number of dimensions", fixed = TRUE)
  # torch does not broadcast the leading dimensions, so a known one carries over from either input
  expect_equal(shapes_out(c(NA, NA, 4L), c(NA, 6L, 5L)), c(NA, 6L, 7L))
})

test_that("shape inference agrees with the module for random shapes and parameters", {
  expect_shape_inference("nn_bilinear", params = function() list(out_features = sample(2:6, 1L)),
    generators = gen_shape(2L), n_in = 2L)
  expect_shape_inference("nn_bilinear", params = function() list(out_features = sample(2:6, 1L)),
    generators = gen_shape(3L), n_in = 2L)
})
