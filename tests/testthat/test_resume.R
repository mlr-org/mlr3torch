make_checkpoint = function(epochs = 2L, freq = 1L, path = tempfile(), callbacks = list(), ...) {
  learner = lrn("classif.mlp", epochs = epochs, batch_size = 50, neurons = 10,
    callbacks = c(list(t_clbk("checkpoint", freq = freq, path = path)), callbacks), ...)
  learner$train(tsk("iris"))
  learner
}

resumer = function(epochs, path, callbacks = list(), ...) {
  learner = lrn("classif.mlp", epochs = epochs, batch_size = 50, neurons = 10,
    callbacks = callbacks, ...)
  learner$param_set$set_values(path = path)
  learner
}

test_that("resuming continues from the checkpointed epoch", {
  path = tempfile()
  make_checkpoint(epochs = 2L, path = path)

  # 'epochs' is the total number of epochs, i.e. 3 more are trained
  resumed = resumer(5L, path)
  resumed$train(tsk("iris"))
  expect_equal(resumed$model$epochs, 5L)
})

test_that("resuming loads the checkpointed weights", {
  path = tempfile()
  first = make_checkpoint(epochs = 2L, path = path)

  # nothing left to train, so the network must be exactly the one of the checkpoint
  resumed = resumer(2L, path)
  expect_warning(resumed$train(tsk("iris")), "No further epochs")

  expect_equal(
    as.numeric(resumed$network$parameters[[1L]]$flatten()),
    as.numeric(first$network$parameters[[1L]]$flatten())
  )
})

test_that("training starts from scratch when there is no checkpoint yet", {
  # the point of this is that the same script can start a run and continue it after a restart
  path = tempfile()
  learner = resumer(2L, path, callbacks = t_clbk("checkpoint", freq = 1, path = path))
  expect_no_error(learner$train(tsk("iris")))
  expect_equal(learner$model$epochs, 2L)

  # the second run of that same script continues where the first one stopped
  again = resumer(4L, path, callbacks = t_clbk("checkpoint", freq = 1, path = path))
  again$train(tsk("iris"))
  expect_equal(again$model$epochs, 4L)
})

test_that("the most recent complete checkpoint is used", {
  path = tempfile()
  make_checkpoint(epochs = 3L, freq = 1L, path = path)
  # a run that died between writing the network and the optimizer
  file.create(file.path(path, "network4.pt"))

  resumed = resumer(4L, path)
  expect_warning(resumed$train(tsk("iris")), "Ignoring incomplete checkpoint\\(s\\) 4")
  # continued from epoch 3, not from the incomplete 4 and not from 1
  expect_equal(resumed$model$epochs, 4L)
})

test_that("checkpoints without a state file are ignored", {
  path = tempfile()
  make_checkpoint(epochs = 2L, freq = 1L, path = path)
  file.remove(file.path(path, "state2.rds"))

  # epoch 2 is incomplete, so the run continues from epoch 1 instead
  resumed = resumer(4L, path)
  expect_warning(resumed$train(tsk("iris")), "Ignoring incomplete checkpoint\\(s\\) 2")
  expect_equal(resumed$model$epochs, 4L)

  # with no complete checkpoint left there is nothing to resume from, and rather than reading the
  # suffixes of a folder written by mlr3torch <= 0.3.3 -- where they may count steps, not epochs --
  # training starts from scratch
  file.remove(list.files(path, pattern = "^state", full.names = TRUE))
  scratch = resumer(2L, path)
  expect_warning(scratch$train(tsk("iris")), "Ignoring incomplete checkpoint")
  expect_equal(scratch$model$epochs, 2L)
})

test_that("the history of the previous run is continued", {
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

test_that("a checkpoint that is already at 'epochs' warns and trains nothing", {
  path = tempfile()
  make_checkpoint(epochs = 3L, path = path, callbacks = list(t_clbk("history")),
    measures_train = msrs("classif.acc"))

  resumed = resumer(3L, path, callbacks = t_clbk("history"), measures_train = msrs("classif.acc"))
  expect_warning(resumed$train(tsk("iris")), "trained for 3 epochs, but 'epochs' is 3")
  expect_equal(resumed$model$epochs, 3L)
  # this run contributes no scores of its own, which must not break the history
  expect_equal(resumed$model$callbacks$history[order(get("epoch"))]$epoch, 1:3)
})

test_that("the learning rate schedule is continued", {
  path = tempfile()
  reference = lrn("classif.mlp", epochs = 4L, batch_size = 50, neurons = 10,
    callbacks = t_clbk("lr_step"))
  reference$param_set$set_values(opt.lr = 0.1, cb.lr_step.step_size = 1, cb.lr_step.gamma = 0.5)
  reference$train(tsk("iris"))

  first = make_checkpoint(epochs = 2L, path = path, callbacks = list(t_clbk("lr_step")),
    opt.lr = 0.1, cb.lr_step.step_size = 1, cb.lr_step.gamma = 0.5)

  resumed = resumer(4L, path, callbacks = t_clbk("lr_step"),
    opt.lr = 0.1, cb.lr_step.step_size = 1, cb.lr_step.gamma = 0.5)
  resumed$train(tsk("iris"))

  # the schedule and the learning rate itself both continue: the schedule comes from the state
  # file, the learning rate from the restored optimizer
  expect_equal(resumed$model$callbacks$lr_step$last_epoch,
    reference$model$callbacks$lr_step$last_epoch)
  expect_equal(resumed$model$optimizer$param_groups[[1L]]$lr,
    reference$model$optimizer$param_groups[[1L]]$lr)
})

test_that("early stopping continues with the score and stagnation of the previous run", {
  path = tempfile()
  args = list(validate = 0.3, measures_valid = msrs("classif.acc"), patience = 5L, min_delta = 0)
  first = invoke(make_checkpoint, epochs = 2L, path = path, .args = args)

  resumed = invoke(resumer, epochs = 4L, path = path, .args = args)
  resumed$train(tsk("iris"))

  # the best epoch of the first run is still a candidate, i.e. the counters were not reset
  expect_true(resumed$model$callbacks$early_stopping$best_epochs >= 1L)
  expect_equal(resumed$model$epochs, 4L)
})

test_that("path = TRUE takes the path from the checkpoint callback", {
  path = tempfile()
  make_checkpoint(epochs = 2L, path = path)

  # the same learner definition serves the first run and every restart
  resumed = lrn("classif.mlp", epochs = 4L, batch_size = 50, neurons = 10, path = TRUE,
    callbacks = t_clbk("checkpoint", freq = 1, path = path))
  resumed$train(tsk("iris"))

  expect_equal(resumed$model$epochs, 4L)
  expect_true(file.exists(file.path(path, "network4.pt")))
})

test_that("path = TRUE errors without a checkpoint callback", {
  learner = lrn("classif.mlp", epochs = 2L, batch_size = 50, neurons = 10, path = TRUE)
  expect_error(learner$train(tsk("iris")), "no 'checkpoint' callback")
})

test_that("path is checked", {
  learner = lrn("classif.mlp", epochs = 1L, batch_size = 50, neurons = 10)
  expect_error(learner$param_set$set_values(path = 1L), "path")
  expect_error(learner$param_set$set_values(path = c("a", "b")), "path")
  expect_error(learner$param_set$set_values(path = FALSE), "path")
  expect_no_error(learner$param_set$set_values(path = NULL))
  expect_no_error(learner$param_set$set_values(path = TRUE))
  expect_no_error(learner$param_set$set_values(path = tempfile()))
})

test_that("callback states that cannot be restored are skipped with a warning", {
  path = tempfile()
  # a callback that saves a state but cannot load one. torch_callback() rejects that combination,
  # so the class is written by hand, as a user extending CallbackSet directly could.
  CallbackSetWriteOnly = R6Class("CallbackSetWriteOnly", inherit = CallbackSet,
    public = list(state_dict = function() list(a = 1)))
  writeonly = as_torch_callback(CallbackSetWriteOnly, id = "writeonly")
  make_checkpoint(epochs = 1L, path = path, callbacks = list(writeonly))

  resumed = resumer(2L, path, callbacks = writeonly)
  expect_warning(resumed$train(tsk("iris")), "does not implement \\$load_state_dict")
})

test_that("states of callbacks that are not part of the run are skipped with a warning", {
  path = tempfile()
  make_checkpoint(epochs = 1L, path = path, callbacks = list(t_clbk("history")))

  resumed = resumer(2L, path)
  expect_warning(resumed$train(tsk("iris")), "'history'.*not part of this training run")
})

test_that("stateless callbacks do not interfere with resuming", {
  path = tempfile()
  noop = torch_callback("noop", on_epoch_end = function() NULL)
  make_checkpoint(epochs = 2L, path = path, callbacks = list(noop))

  resumed = resumer(4L, path, callbacks = noop)
  expect_no_warning(resumed$train(tsk("iris")))
  expect_equal(resumed$model$epochs, 4L)
})

test_that("early stopping still fires in a resumed run", {
  # `stagnation` is restored, so it can start at or above `patience`. An equality test would step
  # over the trigger and never match again, letting the resumed run use its whole epoch budget.
  task = tsk("iris")
  task$internal_valid_task = 1:30
  path = tempfile()
  make = function(epochs, ...) {
    lrn("classif.mlp", epochs = epochs, batch_size = 50, neurons = 10, seed = 1,
      patience = 2L, min_delta = 0, validate = "predefined", measures_valid = msr("classif.ce"),
      opt.lr = 0, ...)
  }

  # opt.lr = 0 means the score can never improve, so the first run stops as soon as it can
  first = make(20L, callbacks = t_clbk("checkpoint", freq = 1, path = path))
  first$train(task)
  expect_equal(first$model$epochs, 3L)
  expect_equal(first$model$callbacks$early_stopping$stagnation, 2L)

  resumed = make(20L, path = path)
  resumed$train(task)
  # it cannot improve either, so it stops at its first validated epoch rather than running to 20
  expect_equal(resumed$model$epochs, 4L)
})

test_that("the history stays ordered by epoch across resumes", {
  path = tempfile()
  make_checkpoint(epochs = 2L, path = path, callbacks = list(t_clbk("history")),
    measures_train = msrs("classif.acc"))

  resumed = resumer(4L, path, callbacks = t_clbk("history"), measures_train = msrs("classif.acc"))
  resumed$train(tsk("iris"))
  expect_equal(resumed$model$callbacks$history$epoch, 1:4)
})

test_that("resuming a checkpoint written by another mlr3torch version warns", {
  path = tempfile()
  make_checkpoint(epochs = 2L, path = path)
  file = file.path(path, "state2.rds")
  state = readRDS(file)
  state$version = "0.0.1"
  saveRDS(state, file)

  resumed = resumer(4L, path)
  expect_warning(resumed$train(tsk("iris")), "written by mlr3torch 0.0.1")
})

test_that("re-running a finished script is a no-op, whatever `freq` is", {
  # `epochs` is the total, so running the same script again resumes a checkpoint that is already
  # there and trains nothing. It must not try to rewrite that checkpoint.
  task = tsk("iris")
  walk(c(1L, 2L, 3L), function(freq) {
    path = tempfile()
    make = function() {
      lrn("classif.mlp", epochs = 5L, batch_size = 50, neurons = 10, path = TRUE,
        callbacks = t_clbk("checkpoint", freq = freq, path = path))
    }
    make()$train(task)
    before = list.files(path)

    again = make()
    expect_warning(again$train(task), "No further epochs are trained")
    expect_equal(again$model$epochs, 5L)
    expect_set_equal(list.files(path), before)
  })
})

test_that("an incomplete checkpoint does not make its folder unusable", {
  # a checkpoint that is too incomplete to resume from must also be one that may be written over,
  # otherwise the folder is stuck: every attempt resumes from an earlier epoch and then collides
  task = tsk("iris")
  path = tempfile()
  first = lrn("classif.mlp", epochs = 6L, batch_size = 50, neurons = 10,
    callbacks = t_clbk("checkpoint", freq = 1, path = path))
  first$train(task)
  file.remove(file.path(path, "optimizer6.pt"))

  resumed = lrn("classif.mlp", epochs = 8L, batch_size = 50, neurons = 10, path = path,
    callbacks = t_clbk("checkpoint", freq = 1, path = path))
  expect_warning(resumed$train(task), "Ignoring incomplete checkpoint\\(s\\) 6")
  expect_equal(resumed$model$epochs, 8L)
})
