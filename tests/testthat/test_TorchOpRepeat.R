test_that("TorchOpRepeat works", {
  task = tsk("mtcars")
  graph = top("input") %>>%
    top("tokenizer") %>>%
    top("linear", out_features = 10) %>>%
    top("relu") %>>%
    top("repeat", times = 2, last = 2) %>>%
    top("head") %>>%
    top("model", optimizer = optim_adam, criterion = nn_mse_loss)
})
