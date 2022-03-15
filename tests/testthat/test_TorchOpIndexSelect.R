test_that("TorchOpIndexSelect works", {
  task = tsk("mtcars")
  block = top("selfattenttion") %>>%
    top("linear")
  graph = top("input") %>>%
    top("tokenizer", d_token = 5) %>>%
    top("repeat", block = block, times = 2) %>>%
    top("indexselect") %>>%
    top("head") %>>%
    top("model", optimizer = optim_adam, criterion = nn_mse_loss)

  output = graph$train(task)

})
