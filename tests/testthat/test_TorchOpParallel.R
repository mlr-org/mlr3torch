test_that("Can easily create skip connection with TorchOps", {
  task = tsk("mtcars")
  block1 = top("input") %>>% top("linear", out_features = 10, id = "linear1") %>>%
    top("relu")
  block2 = top("input") %>>% top("linear", out_features = 10, id = "linear2")
  graph = top("input") %>>%
    top("tokenizer", d_token = 10) %>>%
    top("flatten", start_dim = 2, end_dim = 3) %>>%
    top("parallel", .paths = list(a = block1, b = block2)) %>>%
    top("linear", out_features = 1L) %>>%
    top("model", n_epochs = 0L, optimizer = optim_adam, criterion = nn_mse_loss)

  learner = GraphLearner$new(graph, task_type = "regr")
  learner$train(task)
  # graph$train(task)
})

test_that("extract_paramset works", {
  g1 = Graph$new()$add_pipeop(top("linear", out_features = 2))
  g2 = Graph$new()$add_pipeop(top("linear", out_features = 3))
  paramset = extract_paramset(list(a = g1, b = g2))
  expect_equal(paramset$values, list(a.linear.out_features = 2L, b.linear.out_features = 3L))
})

test_that("nn_parallel works", {
  linear1 = nn_linear(2, 1)
  linear2 = nn_linear(2, 1)
  layer = nn_parallel(linear1, linear2, reduce = "add")
  tensor = torch_randn(10, 2)
  expect_true(torch_equal(linear1(tensor) + linear2(tensor), layer(tensor)))
})
