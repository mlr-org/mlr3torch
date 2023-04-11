test_that("The CallbackTorch class is correct", {
  expect_class(CallbackTorch, "R6ClassGenerator")
  instance = CallbackTorch$new()
  expect_true(is.null(CallbackTorch$inherit))
  expect_true(!inherits(instance, "Callback"))
})

test_that("All stages are called correctly", {
  # We test that:
  # 1. The callbacks are executed in the right order with 1 epoch, 1 training and validation batch
  # 2. Wenn we increase the epoch, training and validation iterations the counts how often the
  # callbacks are executed is correct
  #
  # To implement this, we use a custom callback, that appends its name to a file each time it is
  # executed

  task = tsk("iris")

  write_stage = function(stage) {
    on_stage = function(ctx) {} # nolint
    body(on_stage)[[2L]] = str2lang(sprintf("write(\"%s\", self$path, append = TRUE)", stage))
    on_stage
  }

  cb = torch_callback(id = "test",
    public = list(
      initialize = function(path) {
        assert_path_for_output(path)
        file.create(path)
        self$path = path
      }
    ),
    on_begin = write_stage("on_begin"),
    on_epoch_begin = write_stage("on_epoch_begin"),
    on_after_backward = write_stage("on_after_backward"),
    on_batch_begin = write_stage("on_batch_begin"),
    on_before_valid = write_stage("on_before_valid"),
    on_batch_valid_begin = write_stage("on_batch_valid_begin"),
    on_batch_valid_end = write_stage("on_batch_valid_end"),
    on_batch_end = write_stage("on_batch_end"),
    on_epoch_end = write_stage("on_epoch_end"),
    on_end = write_stage("on_end")
  )
  path = tempfile()
  learner = lrn("classif.mlp", batch_size = 1, epochs = 1, callbacks = cb, cb.test.path = path,
    layers = 0, d_hidden = 1, measures_valid = msr("classif.acc"))
  task$row_roles$use = 2
  task$row_roles$test = 3

  learner$train(task)

  output = readLines(path)
  expect_identical(output, mlr3torch_callback_stages)

  task$row_roles$use = 2:3
  task$row_roles$test = 4:6

  path2 = tempfile()
  learner$param_set$set_values(cb.test.path = path2)
  learner$train(task)
  output2 = readLines(path2)

  check_output = function(output, epochs, ntrain, nvalid) {
    tbl = as.list(table(output))
    train_iter_stages = c("on_after_backward", "on_batch_begin", "on_batch_end")
    valid_iter_stages = c("on_batch_valid_end", "on_batch_valid_begin")
    tbltrain = tbl[train_iter_stages]
    expect_true(unique(unlist(tbltrain)) == ntrain * epochs)

    tblvalid = tbl[valid_iter_stages]
    expect_true(unique(unlist(tblvalid)) == nvalid * epochs)

    stages_once = c("on_begin", "on_end")
    tblonce = tbl[stages_once]
    expect_true(unique(unlist(tblonce)) == 1)

    tblrest = tbl[setdiff(names(tbl), c(train_iter_stages, valid_iter_stages, stages_once))]

  }

  check_output(output2, 1, 2, 3)

  path3 = tempfile()
  learner$param_set$set_values(epochs = 2, cb.test.path = path3)
  learner$train(task)

  output3 = readLines(path3)

  check_output(output3, 2, 2, 3)
})
