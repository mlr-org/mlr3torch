test_that("TorchOpModel works", {
  task = tsk("iris")
  graph = po("pca") %>>%
    top("input") %>>%
    top("tab_tokenizer", d_token = 1L) %>>%
    top("flatten") %>>%
    top("linear_1", out_features = 20L) %>>%
    top("relu_1") %>>%
    top("output") %>>%
    top("loss", loss = "cross_entropy") %>>%
    top("optimizer", optimizer = "adam") %>>%
    top("model.classif",
      batch_size = 16L,
      device = "cpu",
      epochs = 1L,
      callbacks = list()
    )
  glrn = as_learner_torch(graph)
  glrn$id = "net"
  glrn$train(task)
  expect_error(glrn$train(task), regexp = NA)
  expect_error(glrn$predict(task), regexp = NA)

  resampling = rsmp("cv", folds = 2)
  expect_error(resample(task, glrn, resampling), regexp = NA)
})
