test_that("Can train TorchOpLinear", {
  task = make_mtcars_task()
  to_input = TorchOpInput$new()
  to_linear = TorchOpLinear$new(param_vals = list(units = 10))

  graph = Graph$new()
  graph$add_pipeop(to_input)
  graph$add_pipeop(to_linear)
  graph$add_edge(
    src_id = "input",
    src_channel = "architecture",
    dst_id = "linear",
    dst_channel = "architecture"
  )
  graph$add_edge(
    src_id = "input",
    src_channel = "task",
    dst_id = "linear",
    dst_channel = "task"
  )

  graph
  graph = to_input %>>% to_linear
  train_output = graph$train(task)

  expect_identical(task, train_output[[1]])
  expect_r6(train_output[[2]], "Architecture")
})
