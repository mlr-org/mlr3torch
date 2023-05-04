test_that("Basic properties: Classification", {
  expect_pipeop_class(PipeOpTorchModel, constargs = list(task_type = "classif"))

  po_regr = PipeOpTorchModel$new(task_type = "regr")
  expect_pipeop(po_regr)

  po_classif = PipeOpTorchModel$new(task_type = "classif")
  expect_pipeop(po_classif)
})

test_that("Missing configuration gives correct error messages", {
  task = tsk("iris")
  graph = po("torch_ingress_num") %>>% po("nn_head") %>>% po("torch_model_classif")
  expect_error(graph$train(task), regexp = "No loss configured")
  graph1 = po("torch_ingress_num") %>>%
    po("nn_head") %>>%
    po("torch_loss", "cross_entropy") %>>%
    po("torch_model_classif")
  expect_error(graph1$train(task), regexp = "No optimizer configured")
  graph2 = po("torch_ingress_num") %>>%
    po("nn_head") %>>%
    po("torch_loss", "cross_entropy") %>>%
    po("torch_optimizer", "adam") %>>%
    po("torch_model_classif")
  expect_error(graph2$train(task), regexp = "Missing required parameters")
})

test_that("Manual test: Classification and Regression", {
  task = tsk("iris")
  graph = po("torch_ingress_num") %>>%
    po("nn_head") %>>%
    po("torch_loss", "cross_entropy") %>>%
    po("torch_optimizer", "adam")
  md = graph$train(task)[[1L]]
  obj = po("torch_model_classif", epochs = 0, batch_size = 1)
  expect_true(obj$id == "torch_model_classif")

  res = obj$train(list(md))
  expect_equal(res, list(output = NULL))
  expect_class(obj$state, "LearnerClassifTorchModel")
  expect_class(obj$state$model$network, c("nn_graph", "nn_module"))
  # Defaults are used
  expect_class(obj$state$model$optimizer, "optim_adam")
  expect_class(obj$state$model$loss_fn, "nn_crossentropy_loss")

  # It is possible to change parameter values
  md$optimizer = t_opt("adagrad", lr = 0.123)
  obj = po("torch_model_classif", epochs = 0, batch_size = 2)
  obj$train(list(md))
  expect_class(obj$state$model$optimizer, "optim_adagrad")
  expect_true(obj$state$state$param_vals$opt.lr == 0.123)
  expect_true(obj$state$state$param_vals$batch_size == 2)
  # TODO:  Add regr tests

  task = tsk("mtcars")

  graph = po("torch_ingress_num") %>>%
    po("nn_head") %>>%
    po("torch_loss", "cross_entropy") %>>%
    po("torch_optimizer", "adam") %>>%
    po("torch_model_regr",
      batch_size = 10,
      epochs = 1
    )

  graph$train(task)

  pred = graph$predict(task)
  expect_class(pred[[1]], "PredictionRegr")
  learner = graph$pipeops$torch_model_regr$state
  expect_class(learner, "LearnerRegrTorchModel")
})
