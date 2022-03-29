test_that("TorchOpFork works", {
  task = tsk("mtcars")
  block = top("fork", .names = c("a", "b")) %>>%
    gunion(
      list(
        a = top("linear", out_features = 10) %>>% top("relu"),
        b = top("linear", out_features = 10)
      )
    ) %>>%
    top("merge", method = "add", .innum = 2L)
  # block = top("block", .block = block)

  graph = top("tokenizer", d_token = 10) %>>%
    block %>>%
    top("model")
  graph$plot()

  output = graph$train(task)
  architecture = output[[1L]][[2L]]
  debugonce(architecture_reduce)
  architecture_reduce(architecture, task)
})
