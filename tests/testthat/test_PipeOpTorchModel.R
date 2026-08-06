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
  po_model = po("torch_model_regr", epochs = 1L, batch_size = 16,
    measures_valid = msrs(c("regr.mse", "regr.mae")))
  expect_true("validation" %in% po_model$properties)

  graph = po("torch_ingress_num") %>>% po("nn_head") %>>%
    po("torch_loss", "mse") %>>% po("torch_optimizer") %>>% po_model

  glrn = as_learner(graph)
  set_validate(glrn, 0.2)
  expect_equal(glrn$validate, 0.2)
  expect_equal(glrn$graph$pipeops$torch_model_regr$validate, "predefined")
  task = tsk("mtcars")
  glrn$train(task)
  expect_permutation(names(glrn$internal_valid_scores),
    c("torch_model_regr.regr.mse", "torch_model_regr.regr.mae"))
  expect_numeric(glrn$internal_valid_scores$torch_model_regr.regr.mae)
  expect_numeric(glrn$internal_valid_scores$torch_model_regr.regr.mse)

  glrn$param_set$set_values(
    torch_model_regr.measures_valid = list()
  )
  glrn$train(task)
  expect_equal(glrn$internal_valid_scores, named_list())
})

test_that("base_learner works", {
  graph = po("torch_ingress_num") %>>%
    po("nn_head") %>>%
    po("torch_loss", "mse") %>>%
    po("torch_optimizer", "adam") %>>%
    po("torch_model_regr")

  glrn = as_learner(graph)
  expect_equal(glrn$base_learner(return_po = TRUE, recursive = 1)$id, "torch_model_regr")
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
  expect_integerish(glrn$internal_tuned_values$torch_model_regr.epochs)
  glrn$param_set$set_values(torch_model_regr.patience = 0)
  glrn$train(task)
  expect_equal(glrn$internal_tuned_values, named_list())
})

test_that("marshaling", {
  graph = po("torch_ingress_num") %>>%
    po("nn_head") %>>%
    po("torch_loss", "mse") %>>%
    po("torch_optimizer", "adam") %>>%
    po("torch_model_regr", batch_size = 16, epochs = 1L)

  task = tsk("mtcars")
  glrn = as_learner(graph)
  glrn$train(task)
  model = glrn$model
  glrn$marshal()$unmarshal()
  expect_equal(model, glrn$model)
  pred = glrn$predict(task)
  expect_class(pred, "Prediction")
})

test_that("graph-built learners are reproducible from the seed", {
  # `PipeOpTorch` operators build their modules while the Graph trains, i.e. before the learner sets
  # the torch seed, so the weights used to be drawn outside the seeded region. `PipeOpTorchModel`
  # now makes the learner re-initialize them under the seed.
  mk = function(seed) as_learner(
    po("torch_ingress_num") %>>% po("nn_linear", out_features = 4) %>>% po("nn_head") %>>%
      po("torch_loss", t_loss("cross_entropy")) %>>% po("torch_optimizer", t_opt("adam")) %>>%
      po("torch_model_classif", batch_size = 50, epochs = 1, seed = seed, predict_type = "prob"))
  predict_once = function(seed) {
    l = mk(seed)
    l$train(tsk("iris"))
    l$predict(tsk("iris"))$prob
  }

  a = predict_once(1L)
  expect_equal(predict_once(1L), a)
  expect_false(isTRUE(all.equal(predict_once(2L), a)))
})

test_that("a network passed to LearnerTorchModel directly is not re-initialized", {
  # only the pipeop path resets, because a hand-built network's weights may be the point
  task = tsk("mtcars")
  network = nn_linear(task$n_features, 1)
  with_no_grad(network$weight$fill_(0.789))

  learner = LearnerTorchModel$new(network = network, task_type = "regr",
    ingress_tokens = list(input = ingress_num(shape = c(NA, task$n_features))),
    optimizer = t_opt("adam", lr = 0), loss = t_loss("mse"))
  learner$param_set$set_values(batch_size = 32, epochs = 1, seed = 1L)
  learner$train(task)

  expect_equal(unique(as.numeric(learner$model$network$state_dict()[[1L]])), 0.789, tolerance = 1e-6)
})
