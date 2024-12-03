test_that("selectors work", {
  all_params = c("0.weight", "0.bias", "3.weight", "3.bias", "6.weight", "6.bias", "9.weight", "9.bias")

  expect_equal(select_none()(all_params), character(0))
  expect_equal(select_all()(all_params), all_params)
  expect_equal(select_grep("weight")(all_params), c("0.weight", "3.weight", "6.weight", "9.weight"))
  expect_equal(select_invert(select_none())(all_params), all_params)
  expect_equal(select_name("0.bias")(all_params), "0.bias")
})