test_that("Can train TorchOpLinear", {
  task = tsk("mtcars")
  graph = top("input") %>>%
    top("linear", id = "linear1", out_features = 10L) %>>%
    top("relu") %>>%
    top("linear", id = "linear2", out_features = 1L)

  to_input = TorchOpInput$new()
  to_linear = TorchOpLinear$new(param_vals = list(out_features = 10))

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

test_that("TorchOpLinear works with 2D Tensor", {
  task = tsk("mtcars")
  input = torch_randn(16, 7)
  linear = TorchOpLinear$new(
    param_vals = list(out_features = 10)
  )
  layer = linear$build(list(x = input), task)
  output = with_no_grad(layer(input))
  expect_equal(output$shape, c(16, 10))
})

test_that("TorchOpLinear works with 2D Tensor", {
  task = tsk("mtcars")
  input = torch_randn(16, 7, 18)
  linear = TorchOpLinear$new(
    param_vals = list(out_features = 10)
  )
  layer = linear$build(list(x = input), task)
  output = with_no_grad(layer(input))
  expect_equal(output$shape, c(16, 7, 10))
})
