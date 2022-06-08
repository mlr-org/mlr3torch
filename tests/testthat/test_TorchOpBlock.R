test_that("TorchOpBlock works", {
  task = tsk("iris")

  b = top("linear", out_features = 10L) %>>% top("relu")

  graph = top("input") %>>%
    top("tokenizer", d_token = 1L) %>>%
    top("block", .graph = b, times = 2) %>>%
    top("model.classif", epochs = 0L, batch_size = 16L, .optimizer = "adam", .loss = "cross_entropy")


  output = graph$train(task)
  a = output[[1L]]$architecture




})
