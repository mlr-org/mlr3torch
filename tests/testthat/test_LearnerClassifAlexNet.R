test_that("LearnerClasifAlexnet runs", {
  learner = lrn("classif.alexnet", epochs = 10L, batch_size = 1L)
  task = tsk("tiny_imagenet")
  resampling = rsmp("holdout")
  task$row_roles$use = sample(task$nrow, size = 100)
  learner$train(task)
})
