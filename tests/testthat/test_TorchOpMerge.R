test_that("TorchOpMerge works", {
  graph = top("linear", out_features = 10L) %>>%
    top("fork", .names = c("a", "b")) %>>%
    gunion(list(
      a = top("relu") %>>% top("linear", out_features = 10L),
      b = top("linear", out_features = 10L)
    )) %>>%
    top("merge", method = "concat", .input = c("a", "b"))

  parallel(
    a = top("linear") %>>% top("relu"),
    b = top("linear", out_features = 10L)
  )
  gt("parallel",
    a = top("relu") %>>% top("linear"),
    b = top("linear", n)
  )
})
