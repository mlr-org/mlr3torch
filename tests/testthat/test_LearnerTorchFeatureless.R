test_that("dataset_featureless works", {
  task = tsk("iris")
  ds = dataset_featureless(task = task, device = "cpu", target_batchgetter = target_batchgetter("classif"))
  expect_true(ds$.length() == 150)
  batch = ds$.getbatch(1)
  expect_true(torch_equal(batch$x$n, torch_tensor(1L)))
  expect_true(inherits(batch$y, "torch_tensor"))
  expect_equal(batch$.index, 1L)
})

test_that("Basic checks: Classification", {
  learner = lrn("classif.torch_featureless")
  expect_learner_torch(learner)
  expect_set_equal(learner$properties, c("twoclass", "multiclass", "missings", "featureless"))
})

test_that("LearnerTorchFeatureless works", {
  learner = lrn("classif.torch_featureless", batch_size = 50, epochs = 100, seed = 1)
  task = tsk("iris")
  task$row_roles$use = c(1:50, 51:60, 101:110)
  task$row_roles$holdout = 51:150
  learner$train(task)
  pred = learner$predict(task)
  expect_true(pred$response[[1L]] == "setosa")
})


test_that("Basic checks: Regression", {
  learner = lrn("regr.torch_featureless")
  expect_learner_torch(learner)
  expect_set_equal(learner$properties, c("missings", "featureless"))
})
