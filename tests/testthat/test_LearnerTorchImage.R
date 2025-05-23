test_that("LearnerTorchImage works", {
  learner = LearnerTorchImageTest$new(task_type = "classif")
  learner$param_set$set_values(
    epochs = 1,
    batch_size = 1
  )
  task = nano_imagenet()$filter(1)

  expect_equal(learner$man, "mlr3torch::mlr_learners.test")
  expect_r6(learner, c("Learner", "LearnerTorchImage", "LearnerTorch"))
  expect_true(learner$label == "Test Learner Image")
  expect_identical(learner$feature_types, "lazy_tensor")
  expect_set_equal(learner$predict_types, c("response", "prob"))
  expect_subset("R6", learner$packages)

  task = po("trafo_resize", size = c(64, 64))$train(list(task))[[1L]]

  learner$train(task)
  expect_class(learner$network, "nn_module")
  expect_true(is.null(learner$network$modules$`1`$bias))
  pred = learner$predict(task)
  expect_class(pred, "PredictionClassif")
})

test_that("unknown shapes work", {
  learner = lrn("classif.mobilenet_v2", batch_size = 1, epochs = 1)
  task = nano_dogs_vs_cats()$filter(1)
  learner$train(task)
  expect_prediction(learner$predict(task))
})

test_that("single lazy tensor is expected", {
  task = as_task_classif(data.table(
    x1 = as_lazy_tensor(matrix(1:4, 2, 2)),
    x2 = as_lazy_tensor(matrix(1:4, 2, 2)),
    y = as.factor(rep(c("a", "b"), each = 2))
  ), target = "y", id = "test")
  learner = lrn("classif.mobilenet_v2", batch_size = 1, epochs = 1, pretrained = FALSE)
  expect_error(learner$train(task), "expects exactly one feature")
})
