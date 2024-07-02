test_that("Basic properties: Classification", {
  expect_pipeop_class(PipeOpTorchModel, constargs = list(task_type = "classif"))

  po_classif = PipeOpTorchModel$new(task_type = "classif")
  expect_pipeop(po_classif)
})

test_that("Basic properties: Regression", {
  expect_pipeop_class(PipeOpTorchModel, constargs = list(task_type = "regr"))

  po_regr = PipeOpTorchModel$new(task_type = "regr")
  expect_pipeop(po_regr)
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
  expect_class(obj$state, "learner_state")
  expect_class(obj$state$model$network, c("nn_graph", "nn_module"))
  # Defaults are used
  expect_list(obj$state$model$optimizer)
  expect_list(obj$state$model$loss_fn)

  # It is possible to change parameter values
  md$optimizer = t_opt("adagrad", lr = 0.123)
  obj = po("torch_model_classif", epochs = 0, batch_size = 2)
  obj$train(list(md))
  expect_list(obj$state$model$optimizer)
  expect_true(obj$learner_model$optimizer$param_set$values$lr == 0.123)
  expect_true(obj$state$param_vals$batch_size == 2)

  task = tsk("mtcars")

  graph = po("select", selector = selector_name(c("mpg", "cyl"))) %>>%
    po("torch_ingress_num") %>>%
    po("nn_head") %>>%
    po("torch_loss", "mse") %>>%
    po("torch_optimizer", "adam") %>>%
    po("torch_model_regr",
      batch_size = 10,
      epochs = 1
    )

  graph$train(task)

  pred = graph$predict(task)
  expect_class(pred[[1]], "PredictionRegr")
  learner = graph$pipeops$torch_model_regr$state
  expect_class(learner, "learner_state")
})

test_that("phash works", {
  po1 = PipeOpTorchModel$new(task_type = "regr", param_vals = list(shuffle = TRUE))
  po2 = PipeOpTorchModel$new(task_type = "regr", param_vals = list(shuffle = FALSE))
  expect_equal(po1$phash, po2$phash)
  expect_false(
    PipeOpTorchModel$new("regr")$phash == PipeOpTorchModel$new("classif")$phash
  )
})

test_that("validation", {
  po_model = po("torch_model_regr")
  expect_true("validation" %in% po_model$properties)

  graph = po("torch_ingress_num") %>>% po("nn_head") %>>%
    po("torch_loss", "cross_entropy") %>>% po("torch_optimizer") %>>% po_model

  glrn = as_learner(graph)
  set_validate(glrn, 0.2)
  expect_equal(glrn$validate, 0.2)
  expect_equal(glrn$graph$pipeops$torch_model_regr$validate, "predefined")
})


test_that("base_learner works", {
  graph = po("torch_ingress_num") %>>%
    po("nn_head") %>>%
    po("torch_loss", "mse") %>>%
    po("torch_optimizer", "adam") %>>%
    po("torch_model_regr")

  glrn = as_learner(graph)
  expect_equal(glrn$base_learner(return_po = TRUE)$id, "torch_model_regr")
})

test_that("internal_tuning", {
  graph = po("torch_ingress_num") %>>%
    po("nn_head") %>>%
    po("torch_loss", "mse") %>>%
    po("torch_optimizer") %>>%
    po("torch_model_regr", epochs = 1L, batch_size = 3, patience = 10,
      measures_valid = msr("regr.mse"))

  glrn = as_learner(graph)
  glrn$validate = 0.2
  glrn$graph$pipeops$torch_model_regr$validate = "predefined"
  task = tsk("mtcars")
  glrn$train(task)
  expect_true("torch_model_regr.epochs" %in% glrn$internal_tuned_values)
})
