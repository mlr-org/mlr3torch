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

  learner$train(task)

  expect_data_table(learner$model$callbacks$history, nrows = 0)

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

test_that("the history of a previous run is continued when resuming", {
  task = tsk("iris")
  path = tempfile()
  make = function(epochs, callbacks) {
    lrn("classif.mlp", epochs = epochs, batch_size = 50, neurons = 10,
      measures_train = msrs("classif.acc"), callbacks = callbacks)
  }

  learner = make(2L, list(t_clbk("history"), t_clbk("checkpoint", freq = 1)))
  learner$param_set$set_values(cb.checkpoint.path = path)
  learner$train(task)

  resumed = make(4L, list(t_clbk("history")))
  resumed$param_set$set_values(path = path)
  resumed$train(task)

  history = resumed$model$callbacks$history[order(get("epoch"))]
  expect_equal(history$epoch, 1:4)
  # the scores of the first run are the ones that were recorded back then
  expect_equal(
    history[get("epoch") <= 2][["train.classif.acc"]],
    learner$model$callbacks$history[order(get("epoch"))][["train.classif.acc"]]
  )
})
