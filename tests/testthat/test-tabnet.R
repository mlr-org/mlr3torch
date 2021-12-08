# Classification ----------------------------------------------------------
test_that("Learner can be instantiated", {
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

# # FIXME: Figure out what autotest tests and why it fails on feat_all_binary
test_that("autotest", {
  learner = LearnerClassifTorchTabnet$new()
  learner$param_set$values$epochs = 3L
  expect_learner(learner)
  result = run_autotest(learner, exclude = "(feat_single|sanity)", check_replicable = FALSE)
  expect_true(result, info = result$error)
})


# Regression --------------------------------------------------------------
test_that("Learner can be instantiated", {
  skip_if_not_installed("tabnet")

  lrn = LearnerRegrTorchTabnet$new()

  lrn$param_set$values$epochs = 10L
  lrn$param_set$values$decision_width = NULL
  lrn$param_set$values$attention_width = 8L

  expect_learner(lrn)
  expect_identical(lrn$param_set$values$epochs, 10L)
  expect_identical(lrn$param_set$values$attention_width, 8L)
  expect_identical(lrn$param_set$values$decision_width, NULL)
})
