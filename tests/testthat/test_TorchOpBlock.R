test_that("TorchOpBlock works", {
  task = tsk("iris")

  b = top("linear", out_features = 10L) %>>% top("relu")
  op = top("block", .graph = b, times = 1L)
  out = op$build(list(input = torch_randn(16, 3)), task)
  x = out$output$output
  expect_true(all(x$shape == c(16, 10)))


  graph = top("input") %>>%
    top("tokenizer", d_token = 1L) %>>%
    top("flatten") %>>%
    top("block", .graph = b, times = 2) %>>%
    top("loss", .loss = "cross_entropy") %>>%
    top("optimizer", .optimizer = "adam", lr = 0.01) %>>%
    top("model.classif", epochs = 0L, batch_size = 16L)

  expect_error(graph$train(task), regexp = NA)
})
