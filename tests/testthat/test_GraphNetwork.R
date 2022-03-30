test_that("GraphNetwork works", {
  task = tsk("mtcars")
  graph = top("tokenizer", d_token = 3L) %>>%
    top("flatten") %>>%
    top("fork", .outnum = 2L) %>>%
    gunion(
      graphs = list(
        a = top("linear", out_features = 10L),
        b = top("linear", out_features = 10L) %>>% top("relu")
      )
    ) %>>% top("merge", .innum = 2L, method = "stack")

  # %>>%
  # top("merge", method = "add", .innum = 2L) %>>%
  # top("linear", out_features = 1L)
  a = graph$train(task)[[1L]][[2L]]
  res = architecture_reduce(a, task)
  edges = simplify_graph(res$edges)
  dl = make_dataloader(task, batch_size = 16L, device = "cpu")
  batch = dl$.iter()$.next()
  y = batch$y
  batch$y = NULL
  edges$input = list(list())
  debug(network_forward)
  out = network_forward(res$layers, edges, batch, names(res$layers))

})
