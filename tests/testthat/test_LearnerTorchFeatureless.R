test_that("LearnerTorchFeatureless works", {
  learner = lrn("classif.torch_featureless", batch_size = 50, epochs = 100, seed = 1)
  task = tsk("iris")
  task$row_roles$use = c(1:50, 51:60, 101:110)
  task$row_roles$holdout = 51:150
  learner$train(task)
  pred = learner$predict(task)

  expect_true(pred$response[[1L]] == "setosa")
})

