test_that("Basic checks", {
  expect_class(CallbackSet, "R6ClassGenerator")
  instance = CallbackSet$new()
  expect_true(is.null(CallbackSet$inherit))
  expect_true(!inherits(instance, "Callback"))
})

test_that("callback_set is working", {
  expect_subset(mlr_reflections$torch$callback_stages, formalArgs(callback_set))
  expect_subset(formalArgs(callback_set), formalArgs(torch_callback))

  expect_error(callback_set("A"), regexp = "must start with 'CallbackSet'")
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

test_that("weight influences the phash", {
  # two callbacks that differ only in when they are called are not interchangeable
  light = t_clbk("history")
  heavy = t_clbk("history")
  expect_equal(light$phash, heavy$phash)

  heavy$weight = 1
  expect_false(light$phash == heavy$phash)

  # the same weight hashes the same, whether it was set at construction or afterwards
  at_construction = TorchCallback$new(CallbackSetHistory, id = "history", weight = 1)
  afterwards = TorchCallback$new(CallbackSetHistory, id = "history")
  afterwards$weight = 1
  expect_equal(at_construction$phash, afterwards$phash)

  # and clearing it again is not a one-way trip
  heavy$weight = NULL
  expect_equal(light$phash, heavy$phash)
})

test_that("weight reaches the hash of a learner using the callback", {
  # the order the callbacks run in changes what is trained, so two learners that differ only in it
  # must not be treated as the same learner by tuning, benchmarking or caching
  make = function(weight = NULL) {
    cb = t_clbk("history")
    cb$weight = weight
    lrn("classif.mlp", epochs = 1L, batch_size = 50, callbacks = cb)
  }

  expect_equal(make()$hash, make()$hash)
  expect_equal(make(1)$hash, make(1)$hash)

  expect_false(make()$hash == make(1)$hash)
  expect_false(make()$phash == make(1)$phash)
  expect_false(make(1)$hash == make(2)$hash)
})

test_that("callbacks are called in the order they were passed", {
  order = new.env()
  spy = function(id, ...) torch_callback(id,
    on_epoch_end = function() order$seen = c(order$seen, class(self)[[1L]]), ...)

  run = function(callbacks) {
    order$seen = NULL
    lrn("classif.mlp", epochs = 1L, batch_size = 50, neurons = 10,
      callbacks = callbacks)$train(tsk("iris"))
    order$seen
  }

  expect_equal(run(list(spy("a"), spy("b"))), c("CallbackSetA", "CallbackSetB"))
  expect_equal(run(list(spy("b"), spy("a"))), c("CallbackSetB", "CallbackSetA"))
})

test_that("weight overrides the order within a stage", {
  order = new.env()
  spy = function(id, weight = NULL) torch_callback(id, weight = weight,
    on_epoch_end = function() order$seen = c(order$seen, class(self)[[1L]]))

  run = function(callbacks) {
    order$seen = NULL
    lrn("classif.mlp", epochs = 1L, batch_size = 50, neurons = 10,
      callbacks = callbacks)$train(tsk("iris"))
    order$seen
  }

  # a higher weight runs later, whatever the order the callbacks were passed in
  expect_equal(run(list(spy("a", weight = 1), spy("b"))), c("CallbackSetB", "CallbackSetA"))
  expect_equal(run(list(spy("b"), spy("a", weight = 1))), c("CallbackSetB", "CallbackSetA"))
  expect_equal(run(list(spy("a", weight = -1), spy("b"))), c("CallbackSetA", "CallbackSetB"))

  # equal weights keep the order they were passed in
  expect_equal(run(list(spy("a", weight = 2), spy("b", weight = 2))),
    c("CallbackSetA", "CallbackSetB"))
})

test_that("a stateful callback may not run after the checkpoint callback", {
  learner = lrn("classif.mlp", epochs = 2L, batch_size = 50, neurons = 10,
    callbacks = list(t_clbk("checkpoint", freq = 1, path = tempfile()),
      t_clbk("history", weight = Inf)))
  expect_error(learner$train(tsk("iris")), "'history'.*run after the 'checkpoint' callback")

  stateless = lrn("classif.mlp", epochs = 1L, batch_size = 50, neurons = 10,
    callbacks = list(t_clbk("checkpoint", freq = 1, path = tempfile()),
      torch_callback("noop", weight = Inf, on_epoch_end = function() NULL)))
  expect_no_error(stateless$train(tsk("iris")))
})

test_that("a callback's default weight can be overwritten via the TorchCallback", {
  expect_equal(CallbackSetCheckpoint$new(path = tempfile(), freq = 1)$weight, Inf)
  expect_error(CallbackSetCheckpoint$new(path = tempfile(), freq = 1, weight = 3), "unused argument")
  expect_equal(t_clbk("checkpoint", freq = 1, path = tempfile(), weight = 3)$generate()$weight, 3)
  expect_error(t_clbk("checkpoint", freq = 1, path = tempfile(), weight = "high"),
    "Must be of type 'number'")

  expect_equal(CallbackSetEarlyStopping$new(patience = 1, min_delta = 0)$weight, 1000)
  expect_true(CallbackSetEarlyStopping$new(patience = 1, min_delta = 0)$weight <
    CallbackSetCheckpoint$new(path = tempfile(), freq = 1)$weight)
})

test_that("'weight' is reserved and cannot be a parameter", {
  # whether the parameter set is inferred from $initialize() ...
  expect_error(TorchCallback$new(gen), "'weight' is reserved")
  expect_error(TorchCallback$new(gen, param_set = ps(weight = p_dbl(tags = "train"))),
    "'weight' is reserved")
})

test_that("weight is validated", {
  expect_error(torch_callback("bad", weight = "high", on_begin = function() NULL), "weight")
  expect_error(torch_callback("bad", weight = NaN, on_begin = function() NULL), "weight")

  # a class that declares a nonsensical weight is caught before training rather than producing an
  # arbitrary order
  bad = R6::R6Class("CallbackSetBad", inherit = CallbackSet,
    public = list(weight = NaN, on_epoch_end = function() NULL))
  learner = lrn("classif.mlp", epochs = 1L, batch_size = 50, neurons = 10,
    callbacks = list(TorchCallback$new(bad, param_set = ps(), id = "bad")))
  expect_error(learner$train(tsk("iris")), "weight")
})
