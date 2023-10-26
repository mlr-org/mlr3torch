test_that("make_check_vector works", {
  check_vector1 = make_check_vector(1)
  expect_true(check_vector1(1))
  expect_equal(check_vector1(1:2), "Must be an integerish vector of length 1.")

  check_vector2 = make_check_vector(2)
  expect_true(check_vector2(1:2))
  expect_equal(check_vector2(1:3), "Must be an integerish vector of length 1 or 2.")
})

test_that("test_equal_col_info works", {
  ci = data.table(id = "x", type = "factor", levels = list(c("a", "b")))
  expect_true(test_equal_col_info(ci, ci))

  ci1 = data.table(id = "y", type = "factor", levels = list(c("a", "b")))
  expect_false(test_equal_col_info(ci, ci1))
  ci2 = data.table(id = "x", type = "ordered", levels = list(c("a", "b")))
  expect_false(test_equal_col_info(ci, ci2))
  ci3 = data.table(id = "x", type = "factor", levels = list(c("b", "a")))
  expect_false(test_equal_col_info(ci, ci3))
  ci4 = data.table(id = "x", type = "factor", levels = list("a"))
  expect_false(test_equal_col_info(ci, ci4))
})

test_that("get_nout works", {
  expect_equal(get_nout(tsk("iris")), 3)
  expect_equal(get_nout(tsk("mtcars")), 1)
})

test_that("assert_shape works", {
  expect_true(is.null(assert_shape(NULL, null_ok = TRUE)))
  expect_error(assert_shape(NULL, null_ok = FALSE))

  # valid examples
  expect_integer(assert_shape(c(NA, 1)))
  expect_integer(assert_shape(c(NA, 1, 2)))
  expect_integer(assert_shape(c(NA, NA), unknown_ok = TRUE))

  # at least 2 dims
  expect_error(assert_shape(NA))
  # one NA in batch dim
  expect_error(assert_shape(c(1, NA)))
  expect_error(assert_shape(c(NA, NA, unknown_ok = FALSE)))
})

test_that("assert_shapes works", {
  expect_list(assert_shapes(list(c(NA, 1, 2), c(NA, 1)), named = FALSE))
  expect_error(assert_shapes(list(c(NA, 1, 2), c(NA, 1)), named = TRUE))

  expect_list(assert_shapes(list(a = c(NA, 1, 2), b = c(NA, 1)), named = TRUE))
  expect_error(assert_shapes(list(a = NULL, b = c(NA, 1)), named = TRUE))
})

test_that("uniqueify works", {
  expect_equal(uniqueify("a", "a"), "a_1")
})
