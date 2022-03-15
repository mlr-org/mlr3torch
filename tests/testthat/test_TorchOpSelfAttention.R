test_that("TorchOpSelfAttention works", {
  task = tsk("mtcars")
  self_attention = top("tokenizer") %>>%
    top("selfattention") %>>%
    top("linear")


})
