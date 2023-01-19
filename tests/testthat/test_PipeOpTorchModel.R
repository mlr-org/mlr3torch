test_that("PipeOpTorchModel works", {
  graph = po("torch_ingress_num") %>>%
    po("nn_head") %>>%
    po("torch_optimizer", t_opt("sgd"), lr = 0.01) %>>%
    po("torch_loss", t_loss("cross_entropy"))

  graphc = graph$clone()
  graphr = graph$clone()

  graph = graph %>>% po("torch_model", task_type = "regr", epochs = 0)

  graph$train(tsk("iris"))

  graph = po("torch_ingress_num") %>>%
    po("nn_head") %>>%
    po("torch_optimizer", t_opt("sgd"), lr = 0.01) %>>%
    po("torch_loss", t_loss("cross_entropy")) %>>%
    po("torch_model", task_type = "regr")
})


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
