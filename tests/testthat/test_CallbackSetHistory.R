test_that("Autotest", {
  cb = t_clbk("history")
  expect_torch_callback(cb)
})

test_that("CallbackSetHistory works", {
  cb = t_clbk("history")
  task = tsk("iris")
  task$internal_valid_task = task$clone(deep = TRUE)$filter(2)
  task$filter(1)

  learner = lrn("classif.mlp", epochs = 3, batch_size = 1, callbacks = t_clbk("history"), validate = "predefined")

  # without any measures there is nothing for the callback to record, which is an error now rather
  # than an empty table
  expect_error(learner$train(task), "measures_train")

  learner$param_set$set_values(
    measures_train = msrs(c("classif.acc", "classif.ce")),
    measures_valid = msr("classif.ce"))
  learner$train(task)

  expect_equal(colnames(learner$model$callbacks$history), c("epoch", "train.classif.acc", "train.classif.ce", "valid.classif.ce"))
  expect_data_table(learner$model$callbacks$history, nrows = 3)
})

test_that("history works with eval_freq", {
  learner = lrn("regr.torch_featureless", epochs = 10, batch_size = 50, eval_freq = 4, callbacks = "history",
    measures_train = msrs("regr.mse"))
  task = tsk("mtcars")
  learner$train(task)
  expect_equal(learner$model$callbacks$history$epoch, c(4, 8, 10))

  learner$param_set$set_values(eval_freq = 5)
  learner$train(task)
  expect_equal(learner$model$callbacks$history$epoch, c(5, 10))
})

test_that("history errors when no measures are configured", {
  # it only records `measures_train` / `measures_valid`, so without them it used to produce an empty
  # table, which reads like training itself failed
  learner = lrn("classif.mlp", epochs = 2, batch_size = 50, neurons = 5,
    callbacks = t_clbk("history"))
  expect_error(learner$train(tsk("iris")), "measures_train")

  # a configuration error, so it does not trigger the fallback learner
  err = tryCatch(learner$train(tsk("iris")), error = function(e) e)
  expect_class(err, "Mlr3ErrorConfig")

  # either one on its own is enough
  train_only = lrn("classif.mlp", epochs = 2, batch_size = 50, neurons = 5,
    callbacks = t_clbk("history"), measures_train = msr("classif.ce"))
  expect_no_error(train_only$train(tsk("iris")))
  expect_equal(nrow(train_only$model$callbacks$history), 2L)

  valid_only = lrn("classif.mlp", epochs = 2, batch_size = 50, neurons = 5,
    callbacks = t_clbk("history"), validate = 0.3, measures_valid = msr("classif.ce"))
  expect_no_error(valid_only$train(tsk("iris")))
  expect_equal(nrow(valid_only$model$callbacks$history), 2L)
})
