test_that("Can branch architectures", {
  task = tsk("iris")
  graph = top("input") %>>%
    top("linear") %>>%
    top("relu") %>>%
    top("branch", branches = c("a", "b"))

  architecture = graph$train(list(task), FALSE)[[2L]]

})
