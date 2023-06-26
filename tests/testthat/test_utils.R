test_that("utils works", {
  check_vector1 = make_check_vector(1)
  expect_true(check_vector1(1))
  expect_class(check_vector1(1:2), "Must be an integerish vector of length 1.")
})
