test_that("TorchOpModel works", {
  task = tsk("iris")
  graph = po("pca") %>>%
    top("input") %>>%
    top("tokenizer", d_token = 1L) %>>%
    top("flatten") %>>%
    top("linear_1", out_features = 20L) %>>%
    top("relu_1") %>>%
    top("linear_2", out_features = 4L) %>>%
    top("model.classif", .loss = "cross_entropy", .optimizer = "adam", batch_size = 16L,
      device = "cpu", epochs = 1L
    )
  glrn = as_learner(graph)
  glrn$id = "net"
  expect_error(glrn$train(task), regexp = NA)
  expect_error(glrn$predict(task), regexp = NA)

  resampling = rsmp("cv")
  expect_error(resample(task, glrn, resampling), regexp = NA)
})
