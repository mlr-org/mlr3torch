test_that("assert_shape and friends", {
  expect_error(assert_shape("1"))
  expect_error(assert_shape(NULL, null_ok = FALSE))
  expect_error(assert_shape(c(NA, 1), unknown_batch = FALSE))
  expect_error(assert_shape(c(NA, NA), only_batch_unknown = TRUE, unknown_batch = NULL))
  expect_integer(assert_shape(c(NA, NA), only_batch_unknown = FALSE, unknown_batch = NULL))
  expect_integer(assert_shape(c(NA, 1), unknown_batch = TRUE))
  expect_integer(assert_shape(c(NA, 1), unknown_batch = NULL))

  expect_true(is.null(assert_shape(NULL, null_ok = TRUE)))
  expect_integerish(assert_shape(c(1, 2)))
  expect_integerish(assert_shape(c(NA, 2)))
  expect_error(assert_shape(c(2, NA)), regexp = NA)
  expect_error(assert_shape(c(2, NA), unknown_batch = TRUE))
  expect_true(is.integer(assert_shape(c(1, 2), coerce = TRUE)))
  expect_false(is.integer(assert_shape(c(1, 2), coerce = FALSE)))

  expect_error(assert_shapes(list(c(1, 2), c(2, 3)), named = FALSE, unknown_batch = NULL), regexp = NA)
  expect_error(assert_shapes(list(NULL), null_ok = TRUE), regexp = NA)
  expect_error(assert_shapes(list(NULL), null_ok = FALSE))
  expect_error(assert_shapes(list(c(1, 2), c(2, 3)), named = TRUE))
  expect_error(assert_shapes(list(c(1, 2), c(2, 3))), regexp = NA)
  expect_error(assert_shapes(list(c(4, 5), c(2, 3)), unknown_batch = TRUE))
  expect_error(assert_shape(c(NA, 1, 2), len = 2))
  # NULL is ok even when len is specified
  expect_true(check_shape(NULL, null_ok = TRUE, len = 2))
  # NA is valid shape
  expect_true(check_shape(NA))
})

test_that("infer_shapes works", {
  check = function(shapes_in, fn, exp, rowwise = FALSE) {
    if (is.character(exp)) {
      expect_error(infer_shapes(list(x = shapes_in), list(), "y", fn, rowwise, "test"), regexp = exp)
    } else {
      obs = infer_shapes(list(x = shapes_in), list(), "y", fn, rowwise, "test")
      expect_equal(obs[[1L]], exp)
    }
  }

  # general logic
  check(c(NA, 3), identity, c(NA, 3))
  check(c(NA, 3), function(x) x[, -1], NA_integer_)
  check(c(NA, 3), function(x) x[, 1:2], c(NA, 2))
  check(c(NA, NA, 3), function(x) x[, 1:2], c(NA, NA, 3))
  check(c(NA, NA, 3), function(x) x[, 1], c(NA, 3))
  check(c(NA, NA, 3), function(x) x[, 1], c(NA, 3))
  check(c(NA, NA, 3), function(x) x[, 1], c(NA, 3))

  # rowwise
  check(c(10, 4, 3), function(x) x[1, ], c(10, 3), rowwise = TRUE)
  check(c(10, 4, 3), function(x) x[1, ], c(4, 3), rowwise = FALSE)

  # names
  expect_equal(
    names(infer_shapes(list(x = c(NA, 4)), list(), output_names = "out", identity, TRUE, "a")),
    "out"
  )

  # multiple inputs
  expect_equal(
    infer_shapes(list(x = c(NA, 3, 4), y = c(NA, 3)), list(), output_names = c("out1", "out2"), function(x) x[.., 1:2], TRUE, "a"), # nolint
    list(
      out1 = c(NA, 3, 2),
      out2 = c(NA, 2)
    )
  )
  # param_vals
  expect_equal(
    infer_shapes(list(x = c(NA, 4)), fn = function(x, d) x[, d], param_vals = list(d = 1:2), output_names = "out", rowwise = FALSE, "a"), # nolint
    list(
      out = c(NA, 2)
    )
  )
  expect_equal(
    infer_shapes(list(x = c(NA, 4)), fn = function(x, d) x[, d], param_vals = list(d = 1:3), output_names = "out", rowwise = FALSE, "a"), # nolint
    list(
      out = c(NA, 3)
    )
  )

})

test_that("shape-agnostic PipeOps accept unknown non-batch dimensions", {
  # These operators never inspect the unknown dimension when building their module, so
  # requiring every non-batch dimension to be known (the PipeOpTorch default) rejects
  # shapes they can handle perfectly well. See `only_batch_unknown`.
  expect_relaxed = function(id, param_vals, shape, na_idx, n_in = 1L) {
    obj = po(id)
    if (length(param_vals)) obj$param_set$set_values(.values = param_vals)
    testthat::expect_false(get_private(obj)$.only_batch_unknown,
      label = sprintf("%s only_batch_unknown", id))

    shape_na = shape
    shape_na[na_idx] = NA
    shapes_in = rep(list(as.integer(shape_na)), n_in)
    names(shapes_in) = obj$input$name
    shape_out = obj$shapes_out(shapes_in)[[1L]]

    # the module builds from the partially unknown shape and runs on a concrete tensor
    module = get_private(obj)$.make_module(shapes_in, obj$param_set$get_values(), NULL)
    concrete = shape
    concrete[1L] = 2L
    x = rep(list(torch_randn(as.integer(concrete))), n_in)
    out = do.call(module, x)

    # every dimension the pipeop claimed to know must match reality
    expect_equal(length(shape_out), length(dim(out)))
    known = !is.na(shape_out)
    expect_equal(shape_out[known], dim(out)[known],
      label = sprintf("%s known output dims", id))
  }

  # elementwise activations (these six were inconsistent with their 17 siblings)
  expect_relaxed("nn_elu", list(), c(NA, 4, 6), 3)
  expect_relaxed("nn_hardshrink", list(), c(NA, 4, 6), 3)
  expect_relaxed("nn_hardsigmoid", list(), c(NA, 4, 6), 3)
  # gated linear units halve one dimension, NA halves to NA
  expect_relaxed("nn_glu", list(dim = 3), c(NA, 4, 6), 2)
  expect_relaxed("nn_geglu", list(), c(NA, 4, 6), 2)
  expect_relaxed("nn_reglu", list(), c(NA, 4, 6), 2)
  # other shape-preserving operators
  expect_relaxed("nn_dropout", list(p = 0.5), c(NA, 4, 6), 3)
  expect_relaxed("nn_identity", list(), c(NA, 4, 6), 3)
  expect_relaxed("nn_softmax", list(dim = 2), c(NA, 4, 6), 3)
  # merges, including an unknown dimension along the concatenation axis
  expect_relaxed("nn_merge_sum", list(), c(NA, 4, 6), 3, n_in = 2L)
  expect_relaxed("nn_merge_prod", list(), c(NA, 4, 6), 3, n_in = 2L)
  expect_relaxed("nn_merge_cat", list(dim = 2), c(NA, 4, 6), 2, n_in = 2L)
  # pooling never depends on the spatial extent
  expect_relaxed("nn_max_pool1d", list(kernel_size = 2), c(NA, 3, 16), 3)
  expect_relaxed("nn_max_pool2d", list(kernel_size = 2), c(NA, 3, 16, 16), 3)
  expect_relaxed("nn_avg_pool1d", list(kernel_size = 2), c(NA, 3, 16), 3)
  expect_relaxed("nn_avg_pool2d", list(kernel_size = 2), c(NA, 3, 16, 16), 3)
  # reshaping
  expect_relaxed("nn_unsqueeze", list(dim = 2), c(NA, 4, 6), 3)
  expect_relaxed("nn_flatten", list(start_dim = 2, end_dim = 3), c(NA, 4, 6), 3)
  expect_relaxed("nn_reshape", list(shape = c(-1, 24)), c(NA, 4, 6), 3)
})

test_that("adaptive pooling resolves an unknown input dimension to a known output", {
  # The output size is fixed by `output_size`, so an unknown *input* extent still yields a
  # fully known output shape. Rejecting such an input would throw away information.
  for (id in c("nn_adaptive_avg_pool1d", "nn_adaptive_avg_pool2d")) {
    obj = po(id)
    d = if (id == "nn_adaptive_avg_pool1d") 1L else 2L
    obj$param_set$set_values(output_size = rep(4L, d))
    shape_in = c(NA, 3L, rep(NA_integer_, d))
    shape_out = obj$shapes_out(list(input = as.integer(shape_in)))[[1L]]
    expect_equal(shape_out, c(NA, 3L, rep(4L, d)))
  }
})

test_that("PipeOps that need a specific dimension still reject unknown shapes", {
  # Their input shape is (batch, n_features), so the only dimension that could be unknown is
  # the one they read. Relaxing them would gain nothing.
  for (id in c("nn_head", "nn_tokenizer_num", "nn_tokenizer_categ")) {
    expect_true(get_private(po(id))$.only_batch_unknown, label = sprintf("%s stays strict", id))
  }
  expect_error(po("nn_head")$shapes_out(list(c(NA, NA))), "Invalid shape")
})

test_that("PipeOps that read only some dimensions accept unknown shapes elsewhere", {
  # These build their module from a specific dimension but never inspect the others, so only
  # the dimension they read has to be known. See `assert_known_dims()`.
  expect_relaxed = function(id, param_vals, shape, na_idx, task = NULL) {
    obj = po(id)
    if (length(param_vals)) obj$param_set$set_values(.values = param_vals)
    testthat::expect_false(get_private(obj)$.only_batch_unknown,
      label = sprintf("%s only_batch_unknown", id))

    shape_na = as.integer(shape)
    shape_na[na_idx] = NA_integer_
    shapes_in = set_names(list(shape_na), obj$input$name)
    shape_out = obj$shapes_out(shapes_in, task = task)[[1L]]

    # the module builds from the partially unknown shape and runs on a concrete tensor
    module = get_private(obj)$.make_module(shapes_in, obj$param_set$get_values(), task)
    concrete = as.integer(shape)
    concrete[1L] = 2L
    out = module(torch_randn(concrete))

    # every dimension the pipeop claimed to know must match reality
    expect_equal(length(shape_out), length(dim(out)))
    known = !is.na(shape_out)
    expect_equal(shape_out[known], dim(out)[known],
      label = sprintf("%s known output dims", id))
  }

  # convolutions only need the channel dimension, not the spatial extent
  expect_relaxed("nn_conv1d", list(out_channels = 5, kernel_size = 3), c(NA, 3, 17), 3)
  expect_relaxed("nn_conv2d", list(out_channels = 5, kernel_size = 3), c(NA, 3, 17, 19), 3:4)
  expect_relaxed("nn_conv3d", list(out_channels = 5, kernel_size = 3), c(NA, 3, 9, 9, 9), 3:5)
  expect_relaxed("nn_conv_transpose1d", list(out_channels = 5, kernel_size = 3), c(NA, 3, 17), 3)
  expect_relaxed("nn_conv_transpose2d", list(out_channels = 5, kernel_size = 3), c(NA, 3, 17, 19), 3:4)
  # batch norm only needs the feature dimension
  expect_relaxed("nn_batch_norm1d", list(), c(NA, 3, 17), 3)
  expect_relaxed("nn_batch_norm2d", list(), c(NA, 3, 17, 19), 3:4)
  expect_relaxed("nn_batch_norm3d", list(), c(NA, 3, 5, 5, 5), 3:5)
  # layer norm only needs the last `dims` dimensions
  expect_relaxed("nn_layer_norm", list(dims = 1), c(NA, 7, 16), 2)
  expect_relaxed("nn_layer_norm", list(dims = 2), c(NA, 4, 7, 16), 2)
  # the CLS token only needs the token dimension, not the number of tokens
  expect_relaxed("nn_ft_cls", list(initialization = "uniform"), c(NA, 7, 16), 2)
  # squeeze only needs the dimension that is squeezed
  expect_relaxed("nn_squeeze", list(dim = 3), c(NA, 5, 1), 2)
  # linear only needs the last dimension
  expect_relaxed("nn_linear", list(out_features = 3), c(NA, 7, 16), 2)

  # nn_fn does not depend on the shape at all: `infer_shapes()` fills in different values for
  # the unknown dimensions and marks those that vary as unknown again
  obj = po("nn_fn", fn = function(x) x * 2)
  expect_false(get_private(obj)$.only_batch_unknown)
  expect_equal(obj$shapes_out(list(c(NA, NA, 16L)))[[1L]], c(NA, NA, 16L))
})

test_that("relaxed PipeOps give a readable error when the needed dimension is unknown", {
  # Without the check, `NA_integer_` reaches libtorch and fails with a C++ error such as
  # "IntArrayRef contains an int that cannot be represented as a SymInt".
  expect_error(po("nn_conv2d", out_channels = 5, kernel_size = 3)$shapes_out(list(c(NA, NA, 17, 19))),
    "requires the channel dimension (dimension 2) of the input shape to be known", fixed = TRUE)
  expect_error(po("nn_batch_norm2d")$shapes_out(list(c(NA, NA, 17, 19))),
    "requires the feature dimension (dimension 2) of the input shape to be known", fixed = TRUE)
  expect_error(po("nn_layer_norm", dims = 1)$shapes_out(list(c(NA, 7, NA))),
    "requires the last 1 dimension(s), which make up 'normalized_shape', of the input shape", fixed = TRUE)
  expect_error(po("nn_ft_cls")$shapes_out(list(input = c(NA, 7, NA))),
    "requires the token dimension (dimension 3) of the input shape to be known", fixed = TRUE)
  expect_error(po("nn_linear", out_features = 3)$shapes_out(list(c(NA, 7, NA))),
    "requires the last dimension (the number of input features)", fixed = TRUE)
  # without `dim`, squeeze cannot even know the number of output dimensions
  expect_error(po("nn_squeeze")$shapes_out(list(c(NA, NA, 1, 5))),
    "requires all non-batch dimensions (because 'dim' is not specified)", fixed = TRUE)
  expect_error(po("nn_squeeze", dim = 3)$shapes_out(list(c(NA, 5, NA))),
    "requires the squeezed dimension (dimension 3) of the input shape to be known", fixed = TRUE)
})

test_that("nn_block defers to the shape constraints of the PipeOps it wraps", {
  task = tsk("iris")
  block = po("nn_relu") %>>% po("nn_dropout")
  obj = po("nn_block", block, n_blocks = 2)
  # the wrapped PipeOps accept unknown dimensions, so the block must not reject them
  expect_equal(obj$shapes_out(list(c(NA, NA, 16L)), task = task)[[1L]], c(NA, NA, 16L))

  # ... but a wrapped PipeOp that needs a dimension still rejects it
  obj = po("nn_block", po("nn_batch_norm1d"), n_blocks = 1)
  expect_error(obj$shapes_out(list(c(NA, NA, 16L)), task = task),
    "requires the feature dimension", fixed = TRUE)
})

test_that("nn_layer_norm accepts 'dims' up to the number of input dimensions", {
  # the bound used to be the number of input *channels* (always 1), so `dims > 1` failed
  obj = po("nn_layer_norm", dims = 3)
  shape_in = list(c(NA, 4L, 7L, 16L))
  expect_equal(obj$shapes_out(shape_in)[[1L]], c(NA, 4L, 7L, 16L))
  module = get_private(obj)$.make_module(shape_in, obj$param_set$get_values(), NULL)
  expect_equal(unlist(module$normalized_shape), c(4L, 7L, 16L))
  expect_equal(dim(module(torch_randn(2, 4, 7, 16))), c(2, 4, 7, 16))
  # `dims` may not exceed the number of dimensions
  expect_error(po("nn_layer_norm", dims = 5)$shapes_out(shape_in), "dims")
})

test_that("a CNN can be built for images of unknown size", {
  graph = po("nn_conv2d", out_channels = 4, kernel_size = 3) %>>%
    po("nn_batch_norm2d") %>>%
    po("nn_adaptive_avg_pool2d", output_size = c(2, 2)) %>>%
    po("nn_flatten") %>>%
    po("nn_head")

  shape = c(NA, 3L, NA, NA)
  md = ModelDescriptor(
    graph = as_graph(po("nop")),
    ingress = list(nop.input = TorchIngressToken("x", batchgetter_num, shape)),
    task = tsk("iris"),
    pointer = c("nop", "output"),
    pointer_shape = shape
  )
  md_out = graph$train(md)[[1L]]
  # the adaptive pooling resolves the unknown spatial extent to a known output shape
  expect_equal(md_out$pointer_shape, c(NA, 3L))

  network = model_descriptor_to_module(md_out)
  expect_equal(dim(network(torch_randn(2, 3, 11, 13))), c(2, 3))
  expect_equal(dim(network(torch_randn(2, 3, 32, 32))), c(2, 3))
})
