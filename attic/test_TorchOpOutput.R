test_that("TorchOpOutput works", {
  task = tsk("iris")

  out = top("output")
  debugonce(get_private(out)$.train)

  graph = top("input") %>>%
    top("select", items = "num")

  res = graph$train(task)

  out$train(list(input = res[[1L]]))


  de

  b = top("linear", out_features = 10L) %>>% top("relu")
  op = top("repeat", block = b, times = 1L)
  out = op$build(list(input = torch_randn(16, 3)), task)
  x = out$output$output
  expect_true(all(x$shape == c(16, 10)))


    top("repeat", block = b, times = 2) %>>%
    top("loss", loss = "cross_entropy") %>>%
    top("optimizer", optimizer = "adam", lr = 0.01) %>>%
    top("model.classif", epochs = 0L, batch_size = 16L)

  expect_error(graph$train(task), regexp = NA)
})
