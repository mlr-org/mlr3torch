test_that("expect_equal() compares torch objects by value", {
  expect_success(expect_equal(torch_tensor(c(1, 2, 3)), torch_tensor(c(1, 2, 3))))
  expect_failure(expect_equal(torch_tensor(c(1, 2, 3)), torch_tensor(c(100, -5, 7))))
  expect_failure(expect_equal(list(a = torch_tensor(1)), list(a = torch_tensor(2))))
  expect_success(expect_equal(torch_tensor(1), torch_tensor(1.001), tolerance = 0.01))
  # shape and dtype
  expect_failure(expect_equal(torch_tensor(1), torch_tensor(1)$squeeze()))
  expect_failure(expect_equal(torch_tensor(1), torch_tensor(1, dtype = torch_float64())))
  # tensors that require grad
  x = torch_tensor(1, requires_grad = TRUE)
  expect_success(expect_equal(x * 2, x * 2))
  # dtypes and devices
  expect_success(expect_equal(torch_float(), torch_float()))
  expect_failure(expect_equal(torch_float(), torch_long()))
  expect_success(expect_equal(torch_device("cpu"), torch_device("cpu")))
})
