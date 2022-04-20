test_that("nn_rtdl_attention works", {
  d_token = 10
  n_heads = 1
  n_batch = 32
  bias = TRUE
  n_features = 11
  dropout = 0.2
  initialization = "xavier"
  attention = nn_attention(d_token = d_token, n_heads = n_heads, dropout = dropout,
    bias = bias, initialization = initialization
  )
  query = torch_randn(n_batch, n_features, d_token)
  key = torch_randn(n_batch, n_features, d_token)
  output = attention$forward(query, key)
  expect_equal(output$shape, c(n_batch, n_features, d_token))

})
