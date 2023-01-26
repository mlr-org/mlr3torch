test_that("Basic properties", {
  expect_pipeop_class(PipeOpTorchModel, constargs = list(task_type = "classif"))

  po_regr = PipeOpTorchModel$new(task_type = "regr")
  expect_pipeop(po_regr)

  po_classif = PipeOpTorchModel$new(task_type = "classif")
  expect_pipeop(po_classif)

  expect_error(PipeOpTorchModel$new(task_type = "surv"))
})

test_that("Manual test", {
  task = tsk("iris")
  md = (po("torch_ingress_num") %>>% po("nn_head"))$train(task)
  obj = po("torch_model_classif", epochs = 0, batch_size = 1)
  expect_true(obj$id == "torch_model_classif")

  res = obj$train(md)
  expect_equal(res, list(output = NULL))
  expect_class(obj$state, "LearnerClassifTorchModel")
  expect_class(obj$state$model$network, c("nn_graph", "nn_module"))
  # Defaults are used
  expect_class(obj$state$model$optimizer, "optim_adam")
  expect_class(obj$state$model$loss_fn, "nn_crossentropy_loss")

  # It is possible to change values
  md[[1]]$optimizer = t_opt("adagrad", lr = 0.123)
  obj = po("torch_model_classif", epochs = 0, batch_size = 2)
  obj$train(md)
  expect_class(obj$state$model$optimizer, "optim_adagrad")
  expect_true(obj$state$state$param_vals$opt.lr == 0.123)
  expect_true(obj$state$state$param_vals$batch_size == 2)
  # TODO:  Add regr tests
})

test_that("Cloning gives correct error", {
  task = tsk("iris")
  graph = po("torch_ingress_num") %>>% po("nn_head") %>>% po("torch_model_classif", batch_size = 1, epochs = 0)
  graph$train(task)

  expect_error(graph$clone(deep = TRUE), regexp = "Deep clone of trained network is currently not supported.")
})

