test_that("LearnerTorchModel works", {
  # autotest not possible because network is bound to task
  task = tsk("iris")
  learner = LearnerTorchModel$new(
    task_type = "classif",
    network = testmodule_linear(task),
    ingress_tokens = list(x = TorchIngressToken(task$feature_names, batchgetter_num, c(NA, 4L))),
    packages = "data.table"
  )

  learner$param_set$set_values(device = "cpu", epochs = 0, batch_size = 3)

  expect_learner(learner)

  learner$train(task)
  pred = learner$predict(task)
  expect_class(pred, "PredictionClassif")

  expect_equal(
    data.table::address(attr(learner$network, "module")),
    data.table::address(attr(get_private(learner)$.network_stored, "module"))
  )

  expect_set_equal(learner$packages, c("data.table", "mlr3", "mlr3torch", "torch"))
  expect_set_equal(learner$predict_types, names(mlr_reflections$learner_predict_types$classif))
})

test_that("cloning", {
  network = nn_sequential(
    nn_linear(4, 3),
    nn_softmax(dim = 2)
  )
  learner = lrn("classif.torch_model", network = network,
    ingress_tokens = list(x = TorchIngressToken("x", batchgetter_lazy_tensor, c(NA, 4L)))
  )

  learner2 = learner$clone(deep = TRUE)

  learner$param_set$set_values(device = "cpu", epochs = 0, batch_size = 3)
  learner$train(tsk("lazy_iris"))
  expect_error(
    learner$clone(deep = TRUE),
    "because it "
  )
})
