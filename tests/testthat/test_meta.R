test_that("deep clone: tensor", {
  expect_deep_clone(
    torch_tensor(1),
    torch_tensor(1)
  )
  expect_failure(expect_deep_clone(
    torch_tensor(1),
    torch_tensor(2)
  ))
  x = torch_tensor(1)
  expect_error(expect_deep_clone(x, x))

  xgrad = torch_tensor(1, requires_grad = TRUE)
  expect_deep_clone(xgrad, xgrad$clone())

  xbuf = nn_buffer(torch_tensor(1, requires_grad = TRUE))
  expect_deep_clone(xbuf, xbuf$clone())

  xparam = nn_parameter(torch_tensor(1, requires_grad = TRUE))
  expect_deep_clone(xparam, xparam$clone())
})

test_that("deep clone: nn_module", {
  linear = nn_linear(1, 1)
  linear2 = linear$clone(deep = TRUE)
  expect_deep_clone(linear, linear2)
})

test_that("expect_deep_clone for data table", {
  d = data.table(a = 1)
  expect_deep_clone(d, copy(d))
})

test_that("can compare tensors", {
  expect_equal(
    torch_tensor(1)$requires_grad_(FALSE),
    torch_tensor(1)$requires_grad_(FALSE)
  )
  expect_equal(
    torch_tensor(1)$requires_grad_(TRUE),
    torch_tensor(1)$requires_grad_(TRUE)
  )
  expect_failure(expect_equal(
    torch_tensor(1)$requires_grad_(FALSE),
    torch_tensor(1)$requires_grad_(TRUE)
  ))
  expect_failure(expect_equal(
    torch_tensor(1),
    torch_tensor(2)
  ))
})
