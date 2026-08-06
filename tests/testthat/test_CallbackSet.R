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

  # and setting it back to what the dictionary entry declares is not a one-way trip
  heavy$weight = light$weight
  expect_equal(light$phash, heavy$phash)

  # NULL is not that value, it hands the decision back to the CallbackSet class
  afterwards$weight = NULL
  expect_equal(afterwards$generate()$weight, CallbackSetHistory$new()$weight)
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

test_that("the built-in callbacks have the documented weights", {
  # the table in the 'Ordering' section of CallbackSet
  expect_equal(CallbackSet$new()$weight, 0)
  expect_equal(t_clbk("unfreeze")$weight, -200)
  expect_equal(CallbackSetEarlyStopping$new(patience = 1L, min_delta = 0)$weight, 100)
  expect_equal(t_clbk("history")$weight, 200)
  expect_equal(t_clbk("tb")$weight, 300)
  expect_equal(t_clbk("progress")$weight, 400)
  expect_equal(t_clbk("checkpoint", freq = 1, path = tempfile())$generate()$weight, Inf)

  # the schedulers declare it on their base class, so every one of them has it, also the subclasses
  # and the ones a user creates from a torch scheduler
  expect_equal(t_clbk("lr_step", step_size = 1)$generate()$weight, 500)
  expect_equal(t_clbk("lr_one_cycle", max_lr = 0.1)$generate()$weight, 500)
  expect_equal(t_clbk("lr_reduce_on_plateau")$generate()$weight, 500)
  expect_equal(as_lr_scheduler(torch::lr_step, step_on_epoch = TRUE)$generate()$weight, 500)
})

test_that("the built-in callbacks are called in the documented order", {
  seen = new.env()
  # ctx$callbacks is the callbacks in the order they are called in
  spy = torch_callback("spy", on_epoch_end = function() seen$order = names(self$ctx$callbacks))

  learner = lrn("classif.mlp", epochs = 1L, batch_size = 50, neurons = 10,
    validate = 0.3, measures_valid = msr("classif.ce"), patience = 1L,
    callbacks = list(
      # deliberately not in the order they should be called in
      t_clbk("lr_step", step_size = 1),
      t_clbk("checkpoint", freq = 1, path = tempfile()),
      t_clbk("history"),
      spy,
      t_clbk("progress"),
      t_clbk("unfreeze", starting_weights = select_all(), unfreeze = data.table())
    )
  )
  capture.output(capture.output(learner$train(tsk("iris")), type = "message"))

  expect_equal(seen$order,
    c("unfreeze", "spy", "early_stopping", "history", "progress", "lr_step", "checkpoint"))
})

test_that("early stopping decides on the validation scores as the custom callbacks left them", {
  # $on_valid_end() is the stage early stopping acts in, so a callback changing the scores there
  # has to run first for its change to be seen
  cheat = torch_callback("cheat",
    on_valid_end = function() self$ctx$last_scores_valid[[1L]] = self$ctx$epoch)

  learner = lrn("classif.mlp", epochs = 10L, batch_size = 50, neurons = 10, validate = 0.3,
    measures_valid = msr("classif.ce"), patience = 1L, callbacks = cheat)
  learner$train(tsk("iris"))

  # classif.ce is minimized and the callback makes it worse every epoch, so training stops at once
  expect_equal(learner$internal_tuned_values$epochs, 1L)
})

test_that("the lr scheduler steps after the callbacks that report on the epoch", {
  seen = new.env()
  spy = torch_callback("spy",
    on_epoch_end = function() seen$lr = c(seen$lr, self$ctx$optimizer$param_groups[[1L]]$lr))

  lrn("classif.mlp", epochs = 2L, batch_size = 50, neurons = 10, opt.lr = 0.1,
    callbacks = list(t_clbk("lr_step", step_size = 1, gamma = 0.1), spy))$train(tsk("iris"))

  # the learning rate the epoch was trained with, not the one the scheduler stepped to afterwards
  expect_equal(seen$lr, c(0.1, 0.01))
})

test_that("the checkpoint callback runs last, also when another callback has weight Inf", {
  # equal weights otherwise keep the order the callbacks were passed in, which would let this
  # callback change the network after the checkpoint had already saved it
  greedy = torch_callback("greedy", weight = Inf,
    on_epoch_end = function() {
      torch::with_no_grad(self$ctx$network$parameters[[1L]]$mul_(0))
    }
  )

  # the checkpoint holds the zeroed network, i.e. it was written after `greedy` ran
  run = function(callbacks) {
    path = tempfile()
    lrn("classif.mlp", epochs = 1L, batch_size = 50, neurons = 10,
      callbacks = callbacks(path))$train(tsk("iris"))
    as.numeric(torch_load(file.path(path, "network1.pt"))[[1L]]$flatten())
  }
  cp = function(path) t_clbk("checkpoint", freq = 1, path = path)

  expect_true(all(run(function(path) list(cp(path), greedy)) == 0))
  expect_true(all(run(function(path) list(greedy, cp(path))) == 0))
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

test_that("the documented ordering table is generated from the callbacks", {
  tbl = callback_weight_table()
  # not tbl[...] : inside `[` a data.table evaluates `i` with its columns in scope, so `pattern`
  # would resolve to the `name` column rather than to the argument
  weight_of = function(pattern) tbl$weight[grepl(pattern, tbl$name, fixed = TRUE)]

  expect_equal(weight_of("`checkpoint`"), CallbackSetCheckpoint$public_fields$weight)
  expect_equal(weight_of("`history`"), t_clbk("history")$weight)
  expect_equal(weight_of("early stopping"), CallbackSetEarlyStopping$public_fields$weight)
  # every weight has a reason, so adding a level forces one to be written
  expect_false(any(grepl("|  |", callback_weight_section(), fixed = TRUE)))
})
