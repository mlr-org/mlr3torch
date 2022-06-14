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
  network = g$train(task)[[1L]]$network
  expect_r6(network, "nn_Graph")
})


