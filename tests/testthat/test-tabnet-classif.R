test_that("Learner can be instantiated", {
  skip_if_not_installed("tabnet")

  lrn = LearnerClassifTorchTabnet$new()

  lrn$param_set$values$epochs = 10L
  lrn$param_set$values$decision_width = NULL
  lrn$param_set$values$attention_width = 8L

  expect_s3_class(lrn, "R6")
  expect_s3_class(lrn, "LearnerClassif")
  expect_s3_class(lrn, "LearnerClassifTorchTabnet")

  expect_identical(lrn$param_set$values$epochs, 10L)
  expect_identical(lrn$param_set$values$attention_width, 8L)
})
