test_that("Autotest", {
  cb = t_clbk("history")
  expect_torch_callback(cb)
})

test_that("CallbackSetHistory works", {
  cb = t_clbk("history")
  task = tsk("iris")
  task$row_roles$use = 1
  task$row_roles$test = 2

  learner = lrn("classif.mlp", epochs = 3, batch_size = 1, callbacks = t_clbk("history"))

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

test_that("deep clone", {
  history = lrn("classif.torch_featureless", epochs = 0, batch_size = 1, callbacks = "history")$train(tsk("iris"))$
    model$callbacks$history

})
