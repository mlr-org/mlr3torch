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

  expect_data_table(learner$history$train, nrows = 0)
  expect_data_table(learner$history$valid, nrows = 0)

  learner$param_set$set_values(
    measures_train = msrs(c("classif.acc", "classif.ce")),
    measures_valid = msr("classif.ce"))
  learner$train(task)

  expect_equal(colnames(learner$history$train), c("epoch", "classif.acc", "classif.ce"))
  expect_equal(colnames(learner$history$valid), c("epoch", "classif.ce"))

  expect_data_table(learner$history$train, nrows = 3)
  expect_data_table(learner$history$valid, nrows = 3)
})

test_that("plotting works", {
  task = tsk("iris")
  split = partition(task)
  learner = lrn("classif.mlp", epochs = 2, batch_size = 50, callbacks = t_clbk("history"),
    measures_train = msrs(c("classif.acc", "classif.ce", "classif.logloss")), measures_valid = msr("classif.ce"),
    predict_type = "prob"
  )
  task$row_roles$use = split$train
  task$row_roles$test = split$test
  learner$train(task, row_ids = split$train)
  expect_class(learner$history$plot("classif.ce", set = "valid"), "ggplot")
  expect_class(learner$history$plot(c("classif.ce", "classif.acc"), set = "train"), "ggplot")
  learner$history$plot(c("classif.ce", "classif.acc", "classif.logloss"), set = "train")
})
