test_that("TorchOpModel works", {
  task = tsk("iris")
  graph = po("pca") %>>%
    top("input") %>>%
    top("tokenizer", d_token = 3L) %>>%
    top("flatten") %>>%
    top("linear", out_features = 1L) %>>%
    top("model.classif", criterion = "cross_entropy", optimizer = "adam", batch_size = 16L,
      device = "cpu", epochs = 100L
    )
  glrn = as_learner(graph)
  glrn$train(task)
})

test_that("TorchOpModel works", {
  task = tsk("iris")
  graph = po("pca") %>>%
    top("input") %>>%
    top("tokenizer", d_token = 1L) %>>%
    top("flatten") %>>%
    top("linear1", out_features = 10L) %>>%
    top("relu") %>>%
    top("linear2", out_features = 4L) %>>%
    top("model.classif", batch_size = 16L, epochs = 10L,
      criterion = "cross_entropy",
      optimizer = "adam",
      lr = 0.01
    )

  glrn = as_learner(graph)
  resampling = rsmp("holdout")
  rr = resample(task, glrn, resampling)

})
