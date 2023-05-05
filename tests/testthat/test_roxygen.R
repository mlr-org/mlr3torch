test_that("roxy_format works", {
  A = R6Class("A")
  expect_equal(roxy_format(A), "[`R6Class`]")
  B = R6Class("B", inherit = A)
  expect_equal(roxy_format(B), "[`R6Class`] inheriting from [`A`].")
  C = R6Class("C", inherit = B)
  expect_equal(roxy_format(C), "[`R6Class`] inheriting from [`B`], [`A`].")
})
