test_that("LearnerClasifAlexnet runs", {
  learner = lrn("classif.alexnet",
    epochs = 10L,
    batch_size = 1L,
    callbacks = list(cllb("torch.progress")),
    measures = list("acc")
  )
  task = toytask()
  resampling = rsmp("holdout")
  task$row_roles$use = sample(task$nrow, size = 100)
  learner$train(task)
})
