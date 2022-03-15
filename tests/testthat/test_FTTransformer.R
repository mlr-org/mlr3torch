test_that("Can build FT-Transformer using TorchOps", {
  task = tsk("mtcars")
  graph = top("input") %>>%
    top("tokenizer") %>>%
    top("")

})
