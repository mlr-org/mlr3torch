test_that("LearnerClassifTorchImage works", {
  learner = LearnerClassifTorchImageTest$new()
  learner$param_set$values = list(
    height = 64,
    width = 64,
    channels = 3,
    epochs = 1,
    batch_size = 1
  )
  task = tsk("test_imagenet")$filter(1)

  expect_equal(learner$man, "mlr3torch::mlr_learners_classif.test")
  expect_r6(learner, c("LearnerClassif", "LearnerClassifTorchImage", "LearnerClassifTorch"))
  expect_true(learner$label == "Test Learner Image")
  expect_identical(learner$feature_types, "imageuri")
  expect_set_equal(learner$predict_types, c("response", "prob"))
  expect_subset("R6", learner$packages)

  learner$train(task)
  expect_class(learner$network, "nn_module")
  expect_true(is.null(learner$network$modules$`1`$bias))
  pred = learner$predict(task)
  expect_class(pred, "PredictionClassif")
})
