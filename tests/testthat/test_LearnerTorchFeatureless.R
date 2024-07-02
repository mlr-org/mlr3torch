test_that("dataset_featureless works", {
  task = tsk("iris")
  ds = dataset_featureless(task = task, device = "cpu")
  expect_true(ds$.length() == 150)
  batch = ds$.getbatch(1)
  expect_true(torch_equal(batch$x$n, torch_tensor(1L)))
  expect_true(inherits(batch$y, "torch_tensor"))
  expect_equal(batch$.index, torch_tensor(1, torch_long()))
})

test_that("Basic checks: Classification", {
  learner = lrn("classif.torch_featureless", epochs = 1, batch_size = 50)
  expect_learner_torch(learner, task = tsk("iris"))
})

test_that("LearnerTorchFeatureless works", {
  learner = lrn("classif.torch_featureless", batch_size = 50, epochs = 100, seed = 1)
  task = tsk("iris")
  task$row_roles$use = c(1:50, 51:60, 101:110)
  learner$train(task)
  pred = learner$predict(task)
  expect_true(pred$response[[1L]] == "setosa")
})


test_that("Basic checks: Regression", {
  learner = lrn("regr.torch_featureless", epochs = 1, batch_size = 50)
  expect_learner_torch(learner, task = tsk("mtcars"))
})
