test_that("nn_rtdl_attention works", {
  d_token = 10
  n_heads = 2
  n_batch = 32
  bias = TRUE
  dropout = 0.2
  initialization = "xavier"
  attention = nn_attention(d_token = d_token, n_heads = n_heads, dropout = dropout,
    bias = bias, initialization = initialization
  )

})
