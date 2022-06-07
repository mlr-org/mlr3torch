test_that("TorchOpSelfAttention works with single input (selfattention)", {
  task = tsk("mtcars")
  self_attention = top("tokenizer") %>>%
    top("rtdl_attention") %>>%
    top("linear")
})

test_that("TorchOpSelfAttention works with two different inputs", {
  task = tsk("mtcars")
  g = top("input") %>>%
    top("tokenizer", d_token = 3L) %>>%
    gunion(
      graphs = list(
        a = top("linear", out_features = 10L),
        b = top("linear", out_features = 10L)
      )
    ) %>>%
    top("rtdl_attention")
  architecture = g$train(task)[[1L]]$architecture
  network = architecture$build(task)
  expect_r6(network, "nn_Graph")
})


