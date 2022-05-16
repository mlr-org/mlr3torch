test_that("extract_key works", {
  valid_strings = c(
    "linear",
    "linear_1",
    "linear_12111"
  )
  for (string in valid_strings) {
    expect_true(extract_key(string) == "linear")
  }

  invalid_strings = c(
    "linear_12_",
    "linear1"
  )
  for (string in invalid_strings) {
    expect_true(extract_key(string) == string)
  }

  expect_true(extract_key("linear__1") == "linear_")

})
