test_that("LearnerAlexnet runs", {
  learner = lrn("classif.alexnet",
    epochs = 1L,
    batch_size = 1L,
    callbacks = list(),
    optimizer = "adam",
    loss = "cross_entropy",
    pretrained = FALSE
  )
  task = nano_imagenet()
  resampling = rsmp("holdout")
  task$row_roles$use = sample(task$nrow, size = 2)
  learner$train(task)

  pred = learner$predict(task)
  expect_class(pred, "PredictionClassif")
})
