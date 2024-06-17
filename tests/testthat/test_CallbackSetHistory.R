test_that("Autotest", {
  cb = t_clbk("history")
  expect_torch_callback(cb)
})

test_that("CallbackSetHistory works", {
  cb = t_clbk("history")
  task = tsk("iris")
  task$divide(ids = 2)
  task$filter(1)

  learner = lrn("classif.mlp", epochs = 3, batch_size = 1, callbacks = t_clbk("history"), validate = "predefined")

  learner$train(task)

  expect_data_table(learner$model$callbacks$history$train, nrows = 0)
  expect_data_table(learner$model$callbacks$history$valid, nrows = 0)

  learner$param_set$set_values(
    measures_train = msrs(c("classif.acc", "classif.ce")),
    measures_valid = msr("classif.ce"))
  learner$train(task)

  expect_equal(colnames(learner$model$callbacks$history$train), c("epoch", "classif.acc", "classif.ce"))
  expect_equal(colnames(learner$model$callbacks$history$valid), c("epoch", "classif.ce"))

  expect_data_table(learner$model$callbacks$history$train, nrows = 3)
  expect_data_table(learner$model$callbacks$history$valid, nrows = 3)
})

test_that("history works with eval_freq", {
  learner = lrn("regr.torch_featureless", epochs = 10, batch_size = 50, eval_freq = 4, callbacks = "history",
    measures_train = msrs("regr.mse"))
  task = tsk("mtcars")
  learner$train(task)
  expect_equal(learner$model$callbacks$history$train$epoch, c(4, 8, 10))

  learner$param_set$set_values(eval_freq = 5)
  learner$train(task)
  expect_equal(learner$model$callbacks$history$train$epoch, c(5, 10))
})
