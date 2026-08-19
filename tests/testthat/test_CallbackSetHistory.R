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

describe("resuming", {
  it("an epoch that evaluated nothing does not break the appended history", {
    # with eval_freq > 1 such an epoch contributes only the `epoch` column, and the checkpoint asks
    # for the state of every epoch it writes -- which used to error in the middle of writing
    path = tempfile()
    args = list(measures_train = msrs("classif.acc"), eval_freq = 2L)
    first = invoke(make_checkpoint, epochs = 2L, path = path,
      callbacks = list(t_clbk("history")), .args = args)

    resumed = invoke(resumer, epochs = 6L, path = path, callbacks = t_clbk("history"), .args = args)
    expect_no_error(resumed$train(tsk("iris")))

    history = resumed$model$callbacks$history
    # epochs 2, 4 and 6 were evaluated, and the epochs of both runs are in the history
    expect_equal(history$epoch, c(2, 4, 6))
    expect_false(anyNA(history$train.classif.acc))
  })

  it("the history of the previous run is continued", {
    path = tempfile()
    first = make_checkpoint(epochs = 2L, path = path, callbacks = list(t_clbk("history")),
      measures_train = msrs("classif.acc"))

    resumed = resumer(4L, path, callbacks = t_clbk("history"),
      measures_train = msrs("classif.acc"))
    resumed$train(tsk("iris"))

    history = resumed$model$callbacks$history[order(get("epoch"))]
    expect_equal(history$epoch, 1:4)
    # the scores of the first run are the ones that were recorded back then
    expect_equal(
      history[get("epoch") <= 2][["train.classif.acc"]],
      first$model$callbacks$history[order(get("epoch"))][["train.classif.acc"]]
    )
  })

  it("the history stays ordered by epoch across resumes", {
    path = tempfile()
    make_checkpoint(epochs = 2L, path = path, callbacks = list(t_clbk("history")),
      measures_train = msrs("classif.acc"))

    resumed = resumer(4L, path, callbacks = t_clbk("history"), measures_train = msrs("classif.acc"))
    resumed$train(tsk("iris"))
    expect_equal(resumed$model$callbacks$history$epoch, 1:4)
  })
})
