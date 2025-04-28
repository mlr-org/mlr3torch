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
