test_that("Basic checks", {
  expect_class(CallbackSet, "R6ClassGenerator")
  instance = CallbackSet$new()
  expect_true(is.null(CallbackSet$inherit))
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
    on_stage = function() {} # nolint
    body(on_stage)[[2L]] = str2lang(sprintf("write(\"%s\", self$path, append = TRUE)", stage))
    on_stage
  }

  cb = torch_callback(id = "test",
    initialize = function(path) {
      assert_path_for_output(path)
      file.create(path)
      self$path = path
    },
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
  expect_identical(output, mlr_reflections$torch$callback_stages)

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

test_that("callback_set is working", {
  expect_subset(mlr_reflections$torch$callback_stages, formalArgs(callback_set))
  expect_subset(formalArgs(callback_set), formalArgs(torch_callback))

  expect_error(callback_set("A"), regexp = "startsWith")
  tcb = callback_set("CallbackSetA")
  expect_class(tcb, "R6ClassGenerator")
  expect_warning(callback_set("CallbackSetA", public = list(on_edn = function() NULL)), regexp = "on_edn")

  e = new.env()
  e$aaaabbb = 1441
  CallbackSetB = callback_set("CallbackSetB",
    public = list(
      a = 1
    ),
    private = list(
      b = 2
    ),
    active = list(
      c = function() 3
    ),
    parent_env = e
  )
  expect_class(CallbackSetB, "R6ClassGenerator")

  expect_identical(parent.env(CallbackSetB$parent_env), e)
  cb = CallbackSetB$new()
  expect_class(cb, "CallbackSetB")
  expect_identical(cb$a, 1)
  expect_identical(get_private(cb)$b, 2)
  expect_identical(cb$c, 3)

  A = R6Class("A")
  expect_error(callback_set("CallbackSetA", inherit = A), regexp = "does not generate object")
  B = R6Class("B", inherit = CallbackSet)
  expect_error(callback_set("CallbackSetA", inherit = B), regexp = NA)


  CallbackSetC = callback_set("CallbackSetC",
    initialize = function(x) {
      self$x = x
    }
  )

  cb = CallbackSetC$new(1)
  expect_equal(cb$x, 1)

  CallbackSetD = callback_set("CallbackSetD",
    public = list(
      initialize = function(x) {
        self$x = x
      }
    )
  )
  cb = CallbackSetC$new(1)
  expect_equal(cb$x, 1)

  expect_error(
    callback_set("CallbackSetE", public = list(initialize = function() NULL), initialize = function() NULL),
    "initialize"
  )

  CallbackSetF = callback_set("CallbackSetF",
    private = list(deep_clone = function(name, value) value)
  )
  expect_true(CallbackSetF$cloneable)

  CallbackSetG = callback_set("CallbackSetG")
  expect_false(CallbackSetG$cloneable)

  CallbackSetH = callback_set("CallbackSetTestH", initialize = function(ctx) NULL)
  expect_error(TorchCallback$new(CallbackSetH), "is reserved for the ContextTorch")
})


test_that("phash works", {
  expect_equal(t_clbk("checkpoint", freq = 1)$phash, t_clbk("checkpoint", freq = 2)$phash)
  expect_false(t_clbk("history")$phash == t_clbk("progress")$phash)
  expect_false(t_clbk("history", id = "a")$phash == t_clbk("history", id = "b")$phash)
  expect_false(t_clbk("history", label = "a")$phash == t_clbk("history", label = "b")$phash)
})
