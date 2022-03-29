test_that("Architecture is working", {
  task = tsk("mtcars")
  graph = top("input") %>>%
    top("tokenizer", d_token = 1L) %>>%
    top("flatten", start_dim = 2, end_dim = -1L) %>>%
    top("linear", out_features = 10L) %>>%
    top("relu")
  output = graph$train(task)
  a = output[[2]]
  dl = make_dataloader(task, 2, "cpu")
  batch = dl$.iter()$.next()
  net = architecture_reduce(a, task)
  batch$y = NULL

  out = net$forward(batch)

  nodes = make_children(a$nodes)
})

if (FALSE) {
  library(mlr3pipelines)
  library(mlr3)
  graph = po("pca") %>>%
    po("learner", lrn("regr.rpart"))
}

test_that("Architecture is working", {
  task = tsk("mtcars")
  # Graph$debug("train")
  graph = top("tokenizer", d_token = 1L) %>>%
    top("flatten", start_dim = 2, end_dim = -1L) %>>%
    paragraph(
      paths = list(
        a = top("relu") %>>% top("linear", out_features = 20L),
        b = top("linear")
      )
    )
    %>>%
    top("merge", method = "concat") %>>%
    top("linear", id = "head", out_features = 1L)


  output = graph$train(task)
  a = output[[1]][[2]]

  debug(architecture_reduce)
  net = architecture_reduce(a, task)
  batch$y = NULL

  out = net$forward(batch)

  nodes = make_children(a$nodes)
})
