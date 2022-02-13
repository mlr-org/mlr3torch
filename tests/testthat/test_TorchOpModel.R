test_that("Can create simple network", {
  task = make_mtcars_task()
  to_input = TorchOpInput$new()
  to_linear1 = TorchOpLinear$new("linear1", param_vals = list(out_features = 10))
  to_relu1 = TorchOpReLU$new("relu")
  to_linear2 = TorchOpLinear$new("linear2", param_vals = list(out_features = 1))
  to_model = TorchOpModel$new()

  to_model = TorchOpModel$new()
  graph = to_input %>>%
    to_linear1 %>>%
    to_relu1 %>>%
    to_linear2
  output = graph$train(task)

})
