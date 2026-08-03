test_that("PipeOpTorchReshape autotest", {
  obj = po("nn_reshape", shape = c(-1, 2, 2))
  task = tsk("iris")
  graph = po("torch_ingress_num") %>>% obj

  expect_pipeop_torch(graph, "nn_reshape", task)

  # the unknown dimension is resolved, because the number of input elements is known
  out = po("nn_reshape", shape = c(NA, 2, 2))$shapes_out(list(input = c(1, 4)))
  expect_equal(out[[1L]], c(1L, 2L, 2L))
  # ... and stays unknown otherwise
  out = po("nn_reshape", shape = c(NA, 2, 2))$shapes_out(list(input = c(NA, 4)))
  expect_equal(out[[1L]], c(NA, 2L, 2L))
})

test_that("PipeOpTorchReshape paramtest", {
  res = expect_paramset(po("nn_reshape"), nn_reshape)
  expect_paramtest(res)
})

test_that("PipeOpTorchUnsqueeze autotest", {
  obj = po("nn_unsqueeze", dim = 3)
  task = tsk("iris")
  graph = po("torch_ingress_num") %>>% obj

  expect_pipeop_torch(graph, "nn_unsqueeze", task)
})

test_that("PipeOpTorchUnsqueeze paramtest", {
  res = expect_paramset(po("nn_unsqueeze"), nn_unsqueeze)
  expect_paramtest(res)
})

test_that("PipeOpTorchSqueeze autotest", {
  obj = po("nn_squeeze", dim = 3)
  task = tsk("iris")
  graph = po("torch_ingress_num") %>>% po("nn_unsqueeze", dim = 3) %>>%  obj

  x = po("nn_squeeze")

  expect_pipeop_torch(graph, "nn_squeeze", task)
})

test_that("PipeOpTorchSqueeze paramtest", {
  res = expect_paramset(po("nn_unsqueeze"), nn_unsqueeze)
  expect_paramtest(res)
})


test_that("PipeOpTorchFlatten autotest", {
  obj = po("nn_flatten", start_dim = 2, end_dim = 4)
  task = nano_imagenet()
  graph = po("torch_ingress_ltnsr") %>>% obj
  expect_pipeop_torch(graph, "nn_flatten", task)
})

test_that("PipeOpTorchFlatten", {
  res = expect_paramset(po("nn_flatten"), nn_flatten)
  expect_paramtest(res)
})

test_that("nn_unsqueeze interprets negative dim like torch", {
  x = torch_randn(3L, 4L, 6L)
  for (d in c(-1L, -2L, -3L, -4L)) {
    inferred = po("nn_unsqueeze", dim = d)$shapes_out(list(c(NA, 4L, 6L)))[[1L]]
    actual = dim(x$unsqueeze(d))
    expect_equal(length(inferred), length(actual), info = as.character(d))
    # the batch dimension is NA, compare the remaining ones
    expect_equal(inferred[!is.na(inferred)], actual[!is.na(inferred)], info = as.character(d))
  }
  expect_error(po("nn_unsqueeze", dim = 9L)$shapes_out(list(c(NA, 4L, 6L))))
})

test_that("nn_squeeze without dim squeezes all non-batch dimensions", {
  # `dim = NULL` is the documented default and must work both for the inference and the module
  expect_equal(po("nn_squeeze")$shapes_out(list(c(NA, 1L, 4L, 1L)))[[1L]], c(NA, 4L))

  net = nn_squeeze()
  expect_equal(dim(net(torch_randn(3L, 1L, 4L, 1L))), c(3L, 4L))
  # the batch dimension is kept even when it is 1, so that the output matches the inferred shape
  expect_equal(dim(net(torch_randn(1L, 1L, 4L, 1L))), c(1L, 4L))
  expect_equal(dim(nn_squeeze(dim = 2L)(torch_randn(3L, 1L, 4L))), c(3L, 4L))
})

test_that("nn_squeeze removes the same dimensions as the inferred shape for negative dims", {
  # every dimension is removed in one call, so `-1` addresses the last dimension of the input and
  # not the last one of a tensor that a previous removal already shrunk
  expect_equal(po("nn_squeeze", dim = c(-1L, -3L))$shapes_out(list(c(NA, 1L, 5L, 1L)))[[1L]], c(NA, 5L))
  expect_equal(dim(nn_squeeze(dim = c(-1L, -3L))(torch_randn(16L, 1L, 5L, 1L))), c(16L, 5L))
})

test_that("shape inference matches the operator", {
  expect_shapes_out_torch("nn_flatten", list(start_dim = 2, end_dim = 3), c(2, 4, 6, 8))
  expect_shapes_out_torch("nn_reshape", list(shape = c(-1, 24)), c(2, 4, 6))
  expect_shapes_out_torch("nn_unsqueeze", list(dim = 2), c(2, 4, 6))
  expect_shapes_out_torch("nn_unsqueeze", list(dim = -1), c(2, 4, 6))
  expect_shapes_out_torch("nn_squeeze", list(dim = 3), c(2, 4, 1, 6))
  expect_shapes_out_torch("nn_squeeze", list(), c(2, 4, 1, 6))
  # `dim` may select several dimensions
  expect_shapes_out_torch("nn_squeeze", list(dim = c(2L, 3L)), c(4, 1, 1, 8))
  expect_shapes_out_torch("nn_squeeze", list(dim = c(-1L, -2L)), c(4, 1, 1))
  expect_shapes_out_torch("nn_squeeze", list(dim = c(-3L, 4L)), c(4, 1, 8, 1))
  # a duplicated dimension is harmless
  expect_equal(po("nn_squeeze", dim = c(2L, 2L))$shapes_out(list(c(4L, 1L, 1L, 8L)))[[1L]], c(4L, 1L, 8L))
})

test_that("shape inference rejects a 'dim' that does not address a dimension", {
  # assigning to an out-of-range index would extend the shape with `NA`s and build a graph on a
  # shape that no tensor can have
  expect_error(po("nn_unsqueeze", dim = 9L)$shapes_out(list(c(2L, 4L))), "cannot use 'dim' 9", fixed = TRUE)
  expect_error(po("nn_squeeze", dim = -4L)$shapes_out(list(c(2L, 1L, 6L))), "cannot use 'dim' -4", fixed = TRUE)
  # negative values are legal in torch and must be accepted
  expect_equal(po("nn_flatten", end_dim = -2L)$shapes_out(list(c(2L, 4L, 6L, 8L)))[[1L]], c(2L, 24L, 8L))
  # an unknown dimension is assumed to not be 1 and is kept, as for `dim = NULL`
  expect_equal(po("nn_squeeze", dim = 3)$shapes_out(list(c(NA, 5L, NA)))[[1L]], c(NA, 5L, NA))
})

test_that("nn_reshape rejects target dimensions that cannot exist", {
  # such a dimension must not be reported as known and propagated into the rest of the graph
  expect_error(po("nn_reshape", shape = c(-5, -7))$shapes_out(list(c(NA, 4L, 6L))),
    "every dimension must be at least 1", fixed = TRUE)
  expect_error(po("nn_reshape", shape = c(0, 24))$shapes_out(list(c(NA, 4L, 6L))),
    "every dimension must be at least 1", fixed = TRUE)
  # the message echoes the shape the user passed, not the internally rewritten one
  expect_error(po("nn_reshape", shape = c(-1, 7))$shapes_out(list(c(2L, 4L, 6L))), "(-1,7)", fixed = TRUE)

  # torch can only infer one dimension, so the graph must not be built on such a shape at all
  # (`NA` means the same as `-1` here)
  expect_error(po("nn_reshape", shape = c(-1, -1))$shapes_out(list(c(2L, 4L, 6L))),
    "at most one dimension can be inferred", fixed = TRUE)
  expect_error(po("nn_reshape", shape = c(NA, -1, 6))$shapes_out(list(c(NA, 4L, 6L))),
    "at most one dimension can be inferred", fixed = TRUE)
  # ... which also holds when the number of input elements is unknown
  expect_error(po("nn_reshape", shape = c(-1, -1, 6))$shapes_out(list(c(NA, 4L, 6L))),
    "at most one dimension can be inferred", fixed = TRUE)
})

test_that("nn_reshape resolves an unknown target dimension from the input size", {
  # the -1 is resolved whenever the number of input elements is known
  reshape = function(shape, shape_in) po("nn_reshape", shape = shape)$shapes_out(list(shape_in))[[1L]]
  expect_equal(reshape(c(-1, 24), c(32L, 4L, 6L)), c(32L, 24L))
  expect_equal(reshape(c(2, -1), c(32L, 4L, 6L)), c(2L, 384L))
  # ... and keeps it unknown when it is not
  expect_equal(reshape(c(-1, 24), c(NA, 4L, 6L)), c(NA, 24L))
  # a target shape that does not divide the input is rejected
  expect_error(reshape(c(-1, 25), c(32L, 4L, 6L)), "not compatible with the input shape")
  expect_error(reshape(c(32, 25), c(32L, 4L, 6L)), "not compatible with the input shape")
  # an unknown input dimension means the mismatch can only be caught at runtime
  expect_equal(reshape(c(-1, 25), c(NA, 4L, 6L)), c(NA, 25L))
})

test_that("nn_squeeze keeps unknown dimensions instead of rejecting them", {
  # An unknown dimension is assumed to not be 1: rejecting the shape would rule
  # out networks that work at runtime. The module must squeeze exactly the dimensions that
  # `$shapes_out()` squeezed, otherwise the inferred shape and the tensor disagree.
  obj = po("nn_squeeze")
  expect_equal(obj$shapes_out(list(c(NA, NA, 1L, 5L)))[[1L]], c(NA, NA, 5L))
  expect_equal(obj$shapes_out(list(c(NA, NA, 5L)))[[1L]], c(NA, NA, 5L))
  # the batch dimension is never squeezed, also when it is known to be 1
  expect_equal(obj$shapes_out(list(c(1L, 3L, 5L)))[[1L]], c(1L, 3L, 5L))

  # the module agrees with the inferred shape, also when the unknown dimension is 1 at runtime
  shapes_in = list(c(NA, NA, 1L, 5L))
  module = get_private(obj)$.make_module(shapes_in, obj$param_set$get_values(), NULL)
  expect_equal(dim(module(torch_randn(2, 3, 1, 5))), c(2, 3, 5))
  expect_equal(dim(module(torch_randn(2, 1, 1, 5))), c(2, 1, 5))
})

test_that("shape inference agrees with the module for random shapes and parameters", {
  expect_shape_inference_sampled("nn_flatten",
    list(rank = 4L, params = function() list(start_dim = 2L, end_dim = sample(2:3, 1L))))
  expect_shape_inference_sampled("nn_unsqueeze",
    list(rank = 3L, params = function() list(dim = sample(c(1:4, -1L), 1L))))
  expect_shape_inference_sampled("nn_squeeze",
    list(rank = 3L, params = function() list(dim = sample(c(2:3, -1L), 1L)), even = FALSE))
  # the target must match the number of elements per observation, so the shape is not doubled
  expect_shape_inference_sampled("nn_reshape",
    list(rank = 3L, params = function() list(shape = c(-1L, 24L)),
      fixed_shape = c(3L, 4L, 6L), even = FALSE))
})
