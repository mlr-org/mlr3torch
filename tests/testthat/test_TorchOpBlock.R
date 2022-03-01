test_that("Block works", {
  .block = top("linear", out_features = 10) %>>% top("relu")
  block = top("block",
    .block = .block
  )
})

test_that("Block works with repetition", {
  task = tsk("mtcars")
  .block = top("linear", out_features = 10L) %>>%
    top("relu") %>>%
    top("repeat", times = 3L, last = 2L)

  graph = top("input") %>>%
    top("block", .block = .block) %>>%
    top("linear", out_features = 1L)

  architecture = graph$train(task)[[2L]]
})
