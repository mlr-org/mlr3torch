test_that("LearnerTorchModel works", {
  # autotest not possible because network is bound to task
  task = tsk("iris")
  learner = LearnerTorchModel$new(
    task_type = "classif",
    network = testmodule_linear(task),
    packages = "data.table"
  )
  learner$ingress_tokens = list(x = TorchIngressToken(task$feature_names, batchgetter_num, c(NA, 4L)))


  expect_deep_clone(
    learner, learner$clone(deep = TRUE)
  )

  learner$param_set$set_values(device = "cpu", epochs = 0, batch_size = 3)

  expect_learner(learner)

  expect_deep_clone(
    learner, learner$clone(deep = TRUE)
  )
  learner$train(task)
  pred = learner$predict(task)
  expect_class(pred, "PredictionClassif")

  expect_error(learner$train(task), "No network stored")

  expect_set_equal(learner$packages, c("data.table", "mlr3", "mlr3torch", "torch"))
  expect_set_equal(learner$predict_types, names(mlr_reflections$learner_predict_types$classif))
})

test_that("cannot clone trained LearnerTorchModel", {
  # this is impossible, because a LearnerTorchModel is initialized with a network that is then trained
  # Once the learner is trained, the initial state of the network cannot be accessed anymore
  task = tsk("iris")
  learner = LearnerTorchModel$new(
    task_type = "classif",
    network = testmodule_linear(task),
    ingress_tokens = list(x = TorchIngressToken(task$feature_names, batchgetter_num, c(NA, 4L))),
    packages = "data.table",
  )
  learner$param_set$set_values(
    epochs = 0, batch_size = 50
  )

  learner$train(task)
  expect_error(learner$clone(deep = TRUE), "for untrained")
})

test_that("marshaling works for graph learner", {
  graph = po("torch_ingress_num") %>>%
    po("nn_linear", out_features = 20) %>>%
    po("nn_relu") %>>%
    po("nn_head") %>>%
    po("torch_loss", loss = t_loss("cross_entropy")) %>>%
    po("torch_optimizer", optimizer = t_opt("adam", lr = 0.1)) %>>%
    po("torch_callbacks", callbacks = t_clbk("history")) %>>%
    po("torch_model_classif", batch_size = 50, epochs = 1, device = "cpu")

  learner = as_learner(graph)
  learner$id = "graph_mlp"
  task = tsk("iris")
  learner$train(task)
  learner$marshal()
  learner$unmarshal()
  expect_class(learner$predict(task), "Prediction")
})
