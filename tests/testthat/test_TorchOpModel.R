test_that("TorchOpModel works", {
  task = tsk("iris")
  graph = po("pca") %>>%
    top("input") %>>%
    top("tokenizer", d_token = 3L) %>>%
    top("flatten") %>>%
    top("linear", out_features = 4L) %>>%
    top("model.classif", .loss = "cross_entropy", .optimizer = "adam", batch_size = 1L,
      device = "cpu", epochs = 1L
    )
  glrn = as_learner(graph)
  expect_error(glrn$train(task), regexp = NA)
  resampling = rsmp("cv")
  resample(task, glrn, resampling)
})
