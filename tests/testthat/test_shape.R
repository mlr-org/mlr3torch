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
  # These genuinely read the unknown dimension when constructing the module, so the strict
  # default is correct for them and must not be relaxed along with the rest.
  for (id in c("nn_head", "nn_tokenizer_num", "nn_batch_norm1d", "nn_layer_norm", "nn_conv1d")) {
    expect_true(get_private(po(id))$.only_batch_unknown, label = sprintf("%s stays strict", id))
  }
  expect_error(po("nn_batch_norm1d")$shapes_out(list(c(NA, NA, 6))), "Invalid shape")
})
