test_that("nn_cls works", {
  n_batch = 16
  d_token = 7
  n_features = 9
  cls = nn_cls(d_token)
  x = torch_empty(n_batch, n_features, d_token)
  y = cls(x)
  expect_true(all(y$shape == c(16, 10, 7)))
})
