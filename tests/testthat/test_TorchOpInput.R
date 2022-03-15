test_that("Can train TorchOpInput", {
  data = mtcars
  data[["..row_id"]] = seq_len(nrow(data))
  data = as.data.table(data)
  backend = DataBackendTorchDataTable$new(data = data, primary_key = "..row_id")
  task = TaskRegr$new(
    id = "mtcars",
    backend = backend,
    target = "mpg"
  )
  to_input = TorchOpInput$new()
  train_output = to_input$train(list(input = task))
  expect_identical(task, train_output$task)
  expect_r6(train_output$architecture, "Architecture")
})

test_that("Can train TorchOpInput in Graph", {
  data = mtcars
  data[["..row_id"]] = seq_len(nrow(data))
  data = as.data.table(data)
  backend = DataBackendTorchDataTable$new(data = data, primary_key = "..row_id")
  task = TaskRegr$new(
    id = "mtcars",
    backend = backend,
    target = "mpg"
  )
  graph = Graph$new()
  to_input = TorchOpInput$new()
  graph$add_pipeop(to_input)
  train_output = graph$train(task)
  expect_identical(task, train_output$input.task)
  expect_r6(train_output$input.architecture, "Architecture")
})
