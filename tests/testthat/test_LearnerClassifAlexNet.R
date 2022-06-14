test_that("LearnerClassifAlexnet runs", {
  learner = lrn("classif.alexnet",
    epochs = 1L,
    batch_size = 1L,
    callbacks = list(),
    measures = list("acc"),
    .optimizer = "adam",
    .loss = "cross_entropy"
  )
  task = toytask()
  resampling = rsmp("holdout")
  task$row_roles$use = sample(task$nrow, size = 10)
  learner$train(task)
  expect_error(learner$train(task), regexp = NA)
})
