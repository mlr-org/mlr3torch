test_that("assert_shape and friends", {
  expect_error(assert_shape("1"))
  expect_error(assert_shape(NULL, null_ok = FALSE))
  expect_error(assert_shape(c(NA, 1), unknown_batch = FALSE))
  expect_integer(assert_shape(c(NA, NA), unknown_batch = NULL))
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
  # slicing clamps to the input extent, so this is only correct for an input dimension >= 2:
  # the smallest value that unknown dimensions are replaced with is 2, because 1 would be
  # squeezed away and broadcast (see the test on `na_replacements()` below)
  check(c(NA, NA, 3), function(x) x[, 1:2], c(NA, 2, 3))
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

test_that("infer_shapes fills unknown dimensions with values that are not degenerate", {
  check = function(fn, shape, exp) {
    obs = infer_shapes(list(x = shape), list(), "y", fn, rowwise = FALSE, id = "test")[[1L]]
    expect_equal(obs, exp)
  }
  check(function(x) x$squeeze(), c(NA, NA, 6L), c(NA, NA, 6L))
  check(function(x) nnf_conv1d(x, torch_randn(4, 6, 5)), c(NA, 6L, NA), c(NA, 4L, NA))
  check(function(x) x[, 1:32, ], c(NA, NA, 6L), c(NA, NA, 6L))
  check(function(x) x$transpose(2, 3), c(NA, NA, 6L), c(NA, 6L, NA))
})

test_that("na_replacements are lowered when the traced tensor would get too large", {
  # 83^2 * 3 elements is fine, but filling four unknown dimensions with 83 would allocate
  # ~47 million elements per traced tensor
  expect_equal(na_replacements(c(NA, 3L, NA)), c(2L, 83L, 89L))
  expect_true(max(na_replacements(rep(NA_integer_, 4L))) < 83L)
  # no replacement may be 1, whatever the shape looks like
  expect_true(all(na_replacements(rep(NA_integer_, 10L)) > 1L))
  # a shape without unknown dimensions is not traced with different values at all
  expect_equal(na_replacements(c(2L, 3L)), c(2L, 83L, 89L))
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

test_that("shape_to_str formats named shape lists as a single string", {
  # names(x) is a vector, so pasting it against the collapsed shapes would recycle into a
  # character vector and print the error message once per input
  repr = shape_to_str(list(input1 = c(NA, 3L), input2 = c(NA, 5L)))
  expect_string(repr)
  expect_true(grepl("input1", repr, fixed = TRUE))
  expect_true(grepl("input2", repr, fixed = TRUE))
  expect_string(shape_to_str(list(c(NA, 3L), c(NA, 5L))))
  expect_string(shape_to_str(c(NA, 3L)))
})

test_that("infer_shapes reports the error of the function it called", {
  # the error handler must not error itself and swallow the real failure
  res = try(po("nn_fn", fn = function(x) stop("boom"))$shapes_out(list(c(NA, 4L))), silent = TRUE)
  expect_true(inherits(res, "try-error"))
  msg = conditionMessage(attr(res, "condition"))
  expect_true(grepl("boom", msg, fixed = TRUE))
  expect_false(grepl("cannot coerce", msg, fixed = TRUE))
})

test_that("resolve_dim resolves negative indices", {
  shape = c(NA, 3L, 8L, 8L)
  # positive indices are already the true ones
  expect_equal(resolve_dim(2L, shape), 2L)
  # -1 is the last dimension, -length(shape) the first
  expect_equal(resolve_dim(-1L, shape), 4L)
  expect_equal(resolve_dim(-4L, shape), 1L)
  # `dim` may select several dimensions, as for nn_squeeze
  expect_equal(resolve_dim(c(2L, -1L), shape), c(2L, 4L))
  # out-of-range indices stay out of range, so that assert_dim_in_range() reports them
  expect_error(assert_dim_in_range(-5L, resolve_dim(-5L, shape), shape, "po"), "cannot use 'dim'")

  # an operator that inserts a dimension has one more position to address: -1 appends
  expect_equal(resolve_dim(-1L, shape, insert = TRUE), 5L)
  expect_equal(resolve_dim(-5L, shape, insert = TRUE), 1L)
})

test_that("assert_same_batch_size accepts agreeing and unknown batch sizes", {
  # an unknown batch size is compatible with any other, and with another unknown one
  expect_equal(assert_same_batch_size(list(c(NA, 3L), c(NA, 5L)), "po"), NA_integer_)
  expect_equal(assert_same_batch_size(list(c(NA, 3L), c(8L, 5L)), "po"), 8L)
  expect_equal(assert_same_batch_size(list(c(8L, 3L), c(8L, 5L)), "po"), 8L)
  # only the known ones have to agree
  expect_error(assert_same_batch_size(list(c(8L, 3L), c(4L, 5L)), "po"),
    "requires all its inputs to have the same batch size")
})

test_that("assert_ndim accepts several numbers of dimensions", {
  expect_silent(assert_ndim(c(NA, 3L), 2L, "po"))
  expect_silent(assert_ndim(c(NA, 3L), c(2L, 3L), "po"))
  expect_silent(assert_ndim(c(NA, 3L, 4L), c(2L, 3L), "po"))
  # the message lists all accepted values
  expect_error(assert_ndim(c(NA, 3L, 4L, 5L), c(2L, 3L), "po"),
    "requires an input with 2 or 3 dimensions", fixed = TRUE)
  expect_error(assert_ndim(c(NA, 3L, 4L), 2L, "po"),
    "requires an input with 2 dimensions", fixed = TRUE)
})

test_that("assert_ndim accepts min and max bounds", {
  expect_silent(assert_ndim(c(NA, 3L), id = "po", min = 2L, max = 3L))
  expect_silent(assert_ndim(c(NA, 3L, 4L), id = "po", min = 2L, max = 3L))
  # an adjacent range reads as "or", a wider one as a range
  expect_error(assert_ndim(c(NA, 3L, 4L, 5L), id = "po", min = 2L, max = 3L),
    "requires an input with 2 or 3 dimensions", fixed = TRUE)
  expect_error(assert_ndim(c(NA, 3L, 4L, 5L, 6L), id = "po", min = 2L, max = 4L),
    "requires an input with 2 to 4 dimensions", fixed = TRUE)
  expect_error(assert_ndim(c(NA, 3L, 4L), id = "po", min = 4L, max = 4L),
    "requires an input with 4 dimensions", fixed = TRUE)

  # each bound also works on its own
  expect_silent(assert_ndim(c(NA, 3L, 4L, 5L), id = "po", min = 3L))
  expect_error(assert_ndim(c(NA, 3L), id = "po", min = 3L),
    "requires an input with at least 3 dimensions", fixed = TRUE)
  expect_silent(assert_ndim(c(NA, 3L), id = "po", max = 3L))
  expect_error(assert_ndim(c(NA, 3L, 4L, 5L), id = "po", max = 3L),
    "requires an input with at most 3 dimensions", fixed = TRUE)
})

test_that("assert_shape reports its own message for non-shapes", {
  # a value that `shape_to_str()` cannot format must not leak its internal assertion
  expect_error(assert_shape("not a shape"), "Invalid shape: must be an integer vector, but is character", fixed = TRUE)
  expect_error(assert_shape(sum), "Invalid shape: must be an integer vector, but is function", fixed = TRUE)
  # the ones it can format still show the shape
  expect_error(assert_shape(list(1, 2)), "Invalid shape: [(1);(2)]", fixed = TRUE)
  expect_error(assert_shape(integer(0)), "Invalid shape: ()", fixed = TRUE)
})
