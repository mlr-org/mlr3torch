test_that("assert_shape works", {
  expect_error(assert_shape("1"))
  expect_error(assert_shape(NULL, null_ok = FALSE))

  expect_true(is.null(assert_shape(NULL, null_ok = TRUE)))
  expect_integerish(assert_shape(c(1, 2)))
  expect_integerish(assert_shape(c(NA, 2)))
  expect_error(assert_shape(c(2, NA)), regexp = "Invalid")
  expect_true(is.integer(assert_shape(c(1, 2), coerce = TRUE)))
  expect_false(is.integer(assert_shape(c(1, 2), coerce = FALSE)))
})
