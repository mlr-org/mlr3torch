# Classification ----------------------------------------------------------
test_that("LearnerClassifTorchTabnet can be instantiated", {
  skip_if_not_installed("tabnet")

  lrn = LearnerClassifTorchTabnet$new()

  lrn$param_set$values$epochs = 10L
  lrn$param_set$values$decision_width = NULL
  lrn$param_set$values$attention_width = 8L

  expect_learner(lrn)
  expect_identical(lrn$param_set$values$epochs, 10L)
  expect_identical(lrn$param_set$values$attention_width, 8L)
  expect_identical(lrn$param_set$values$decision_width, NULL)
})

test_that("LearnerClassifTorchTabnet autotest", {
  learner = LearnerClassifTorchTabnet$new()
  learner$param_set$values$epochs = 1L
  learner$param_set$values$num_threads = 1L
  expect_learner(learner)
  result = run_autotest(learner, exclude = "(feat_single|sanity)", check_replicable = FALSE)
  expect_true(result, info = result$error)
})


# Regression --------------------------------------------------------------
test_that("LearnerRegrTorchTabnet can be instantiated", {
  skip_if_not_installed("tabnet")

  lrn = LearnerRegrTorchTabnet$new()

  lrn$param_set$values$epochs = 1L
  lrn$param_set$values$decision_width = NULL
  lrn$param_set$values$attention_width = 8L

  expect_learner(lrn)
  expect_identical(lrn$param_set$values$epochs, 10L)
  expect_identical(lrn$param_set$values$attention_width, 8L)
  expect_identical(lrn$param_set$values$decision_width, NULL)
})

test_that("LearnerRegrTorchTabnet autotest", {
  learner = LearnerRegrTorchTabnet$new()
  learner$param_set$values$epochs = 1L
  learner$param_set$values$num_threads = 1L
  expect_learner(learner)
  result = run_autotest(learner, exclude = "(feat_single|sanity)", check_replicable = FALSE)
  expect_true(result, info = result$error)
})
