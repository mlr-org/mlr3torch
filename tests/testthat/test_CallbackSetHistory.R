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

test_that("a restored history is continued and not prepended", {
  # the state dict is only loaded by a callback of the user's own, so a minimal one does it here
  # distinct epoch numbers, so that the ordering and not only the contents is pinned down
  state = data.table(epoch = c(7, 8), train.regr.mse = c(100, 90))
  loader = torch_callback("loader",
    on_begin = function() self$ctx$callbacks$history$load_state_dict(state)
  )
  learner = lrn("regr.torch_featureless", epochs = 2, batch_size = 50,
    callbacks = list(t_clbk("history"), loader), measures_train = msrs("regr.mse"))
  learner$train(tsk("mtcars"))

  history = learner$model$callbacks$history
  expect_equal(history$epoch, c(7, 8, 1, 2))
  expect_equal(history$train.regr.mse[1:2], c(100, 90))
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
