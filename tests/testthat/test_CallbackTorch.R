test_that("Basic checks", {
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
    on_stage = function() {} # nolint
    body(on_stage)[[2L]] = str2lang(sprintf("write(\"%s\", self$path, append = TRUE)", stage))
    on_stage
  }

  cb = callback_descriptor(id = "test",
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

test_that("callback_torch is working", {
  expect_subset(mlr3torch_callback_stages, formalArgs(callback_torch))
  expect_subset(formalArgs(callback_torch), formalArgs(callback_descriptor))

  expect_error(callback_torch("A"), regexp = "startsWith")
  tcb = callback_torch("CallbackTorchA")
  expect_class(tcb, "R6ClassGenerator")
  expect_warning(callback_torch("CallbackTorchA", public = list(on_edn = function() NULL)), regexp = "on_edn")

  e = new.env()
  e$aaaabbb = 1441
  CallbackTorchB = callback_torch("CallbackTorchB",
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
  expect_class(CallbackTorchB, "R6ClassGenerator")

  expect_identical(parent.env(CallbackTorchB$parent_env), e)
  cb = CallbackTorchB$new()
  expect_class(cb, "CallbackTorchB")
  expect_identical(cb$a, 1)
  expect_identical(get_private(cb)$b, 2)
  expect_identical(cb$c, 3)

  A = R6Class("A")
  expect_error(callback_torch("CallbackTorchA", inherit = A), regexp = "does not generate object")
  B = R6Class("B", inherit = CallbackTorch)
  expect_error(callback_torch("CallbackTorchA", inherit = B), regexp = NA)


  CallbackTorchC = callback_torch("CallbackTorchC",
    initialize = function(x) {
      self$x = x
    }
  )

  cb = CallbackTorchC$new(1)
  expect_equal(cb$x, 1)

  CallbackTorchD = callback_torch("CallbackTorchD",
    public = list(
      initialize = function(x) {
        self$x = x
      }
    )
  )
  cb = CallbackTorchC$new(1)
  expect_equal(cb$x, 1)

  expect_error(
    callback_torch("CallbackTorchE", public = list(initialize = function() NULL), initialize = function() NULL),
    "initialize"
  )

  CallbackTorchF = callback_torch("CallbackTorchF",
    private = list(deep_clone = function(name, value) "cloning works")
  )
  expect_true(CallbackTorchF$cloneable)
  cbf = CallbackTorchF$new()
  expect_equal(get_private(cbf)$deep_clone("a", 1), "cloning works")

  CallbackTorchG = callback_torch("CallbackTorchG")
  expect_false(CallbackTorchG$cloneable)

  CallbackTorchH = callback_torch("CallbackTorchTestH", initialize = function(ctx) NULL)
  expect_error(DescriptorTorchCallback$new(CallbackTorchH), "is reserved for the ContextTorch")
})
