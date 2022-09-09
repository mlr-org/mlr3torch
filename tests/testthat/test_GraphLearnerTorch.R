test_that("GraphLearnerTorch works", {
  g = top("input") %>>%
    top("select", items = "num") %>>%
    top("linear", out_features = 10L) %>>%
    top("relu") %>>%
    top("output") %>>%
    top("optimizer", "adam", lr = 0.1) %>>%
    top("loss", "cross_entropy")

  g_classif = g %>>% top("model.classif", epochs = 1L, batch_size = 16L)
  g_regr = g %>>% top("model.regr", epochs = 1L, batch_size = 16L)

  lrn_classif = as_learner_torch(g_classif)
  assert_true(lrn_classif$task_type == "classif")
  task = tsk("iris")
  lrn_classif$train(task)
  expect_class(lrn_classif$network, "nn_graph")
  expect_class(lrn_classif$history, "History")
  expect_class(lrn_classif$optimizer, "torch_optimizer")
  expect_class(lrn_classif$loss_fn, "nn_loss")

  lrn_regr = as_learner_torch(g_regr)
  assert_true(lrn_regr$task_type == "regr")
})
