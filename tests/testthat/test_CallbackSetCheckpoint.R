test_that("Autotest", {
  cb = t_clbk("checkpoint", freq = 1, path = tempfile())
  expect_torch_callback(cb)
})

test_that("manual", {
  cb = t_clbk("checkpoint", freq = 1)
  task = tsk("iris")
  task$row_roles$use = 1

  pth0 = tempfile()
  learner = lrn("classif.mlp", epochs = 3, batch_size = 1, callbacks = cb)
  learner$param_set$set_values(cb.checkpoint.path = pth0)

  learner$train(task)

  expect_set_equal(
    c(paste0("network", 1:3, ".pt"), paste0("optimizer", 1:3, ".pt"), paste0("state", 1:3, ".rds")),
    list.files(pth0)
  )


  learner = lrn("classif.mlp", epochs = 3, batch_size = 1, callbacks = cb)
  pth2 = tempfile()
  learner$param_set$set_values(cb.checkpoint.path = pth2, cb.checkpoint.freq = 2)
  learner$train(task)

  expect_set_equal(
    c("network2.pt", "optimizer2.pt", "state2.rds", "network3.pt", "optimizer3.pt", "state3.rds"),
    list.files(pth2)
  )
  pred = learner$predict(tsk("iris"))

  opt_state = torch_load(file.path(pth2, "optimizer3.pt"))
  expect_list(opt_state, types = c("numeric", "list", "torch_tensor"))
})

test_that("error when using an existing directory that holds unrelated data", {
  path = tempfile()
  dir.create(path)
  writeLines("not a checkpoint", file.path(path, "notes.txt"))
  cb = t_clbk("checkpoint", freq = 1, path = path)
  expect_error(cb$generate(), "already exists")
})

test_that("an existing empty directory can be checkpointed into", {
  # a pre-created output folder, and what a run that died before its first checkpoint leaves behind
  path = tempfile()
  dir.create(path)
  expect_no_error(t_clbk("checkpoint", freq = 1, path = path)$generate())

  task = tsk("iris")
  learner = lrn("classif.mlp", epochs = 2L, batch_size = 50, neurons = 10,
    callbacks = t_clbk("checkpoint", freq = 1, path = path))
  expect_no_error(learner$train(task))
  expect_set_equal(list.files(path),
    c(paste0("network", 1:2, ".pt"), paste0("optimizer", 1:2, ".pt"), paste0("state", 1:2, ".rds")))
})

test_that("an epoch that failed is not saved under its own number", {
  task = tsk("iris")
  path = tempfile()
  # crashes in the middle of epoch 3, after epoch 2 was saved by frequency
  crash = torch_callback("Crash",
    on_batch_end = function() if (self$ctx$epoch == 3L && self$ctx$step == 2L) stop("crash"))
  learner = lrn("classif.mlp", epochs = 6L, batch_size = 50, neurons = 10,
    callbacks = list(t_clbk("checkpoint", freq = 2, path = path), crash))
  expect_error(learner$train(task), "crash")

  # 'network<n>.pt' is the network at the end of epoch n, and epoch 3 never reached its end, so
  # only what `freq` wrote at the end of epoch 2 remains
  expect_set_equal(list.files(path), c("network2.pt", "optimizer2.pt", "state2.rds"))

  # the final epoch is still stored when training completes normally, also when `freq` skips it
  path2 = tempfile()
  done = lrn("classif.mlp", epochs = 3L, batch_size = 50, neurons = 10,
    callbacks = t_clbk("checkpoint", freq = 2, path = path2))
  done$train(task)
  expect_set_equal(list.files(path2),
    c("network2.pt", "optimizer2.pt", "state2.rds", "network3.pt", "optimizer3.pt", "state3.rds"))
})

test_that("the state file holds the epoch and the states of the other callbacks", {
  task = tsk("iris")
  path = tempfile()
  learner = lrn("classif.mlp", epochs = 2L, batch_size = 50, neurons = 10,
    measures_train = msrs("classif.acc"),
    callbacks = list(t_clbk("checkpoint", freq = 1, path = path), t_clbk("history")))
  learner$train(task)

  state = readRDS(file.path(path, "state2.rds"))
  expect_equal(state$epoch, 2L)
  expect_names(names(state$callbacks), must.include = "history")
  # saveRDS() rather than torch_save(), which would strip the data.table class off the history
  expect_data_table(state$callbacks$history)
  expect_equal(state$callbacks$history, learner$model$callbacks$history)

  # the checkpoint callback itself is stateless and therefore not part of the file
  expect_true("checkpoint" %nin% names(state$callbacks))
})

test_that("a failure writes nothing beyond what `freq` had already saved", {
  task = tsk("iris")
  path = tempfile()
  # crashes in the middle of epoch 3, which `freq` would not have saved anyway
  crash = torch_callback("crash",
    on_batch_end = function() if (self$ctx$epoch == 3L && self$ctx$step == 2L) stop("crash"))
  learner = lrn("classif.mlp", epochs = 6L, batch_size = 50, neurons = 10,
    callbacks = list(t_clbk("checkpoint", freq = 5, path = path), crash))
  expect_error(learner$train(task), "crash")

  # epoch 2 is the last complete one, but the batches of epoch 3 that ran have already updated the
  # network and the optimizer, so there is nothing left to write that is the end of an epoch
  expect_equal(list.files(path), character(0))
})

test_that("a checkpoint is never a half-trained epoch under the previous epoch's number", {
  task = tsk("iris")
  path = tempfile()
  # the weights at the end of epoch 2, to compare the checkpoint against
  weights = NULL
  spy = torch_callback("spy", on_epoch_end = function() {
    if (self$ctx$epoch == 2L) weights <<- as.numeric(self$ctx$network$parameters[[1L]]$flatten())
  })
  crash = torch_callback("crash",
    on_batch_end = function() if (self$ctx$epoch == 3L && self$ctx$step == 2L) stop("crash"))

  # freq = 2 saves epoch 2 before the crash, and nothing may overwrite it with the half-trained
  # weights of epoch 3
  learner = lrn("classif.mlp", epochs = 6L, batch_size = 30, neurons = 10,
    callbacks = list(t_clbk("checkpoint", freq = 2, path = path), spy, crash))
  expect_error(learner$train(task), "crash")

  saved = as.numeric(torch_load(file.path(path, "network2.pt"))[[1L]]$flatten())
  expect_equal(saved, weights)
})

test_that("ending a run early still saves the epoch that finished", {
  task = tsk("iris")
  path = tempfile()
  # `ctx$terminate` is only acted on after `on_epoch_end`, unlike an error inside an epoch
  stopper = torch_callback("stopper",
    on_epoch_end = function() if (self$ctx$epoch == 3L) self$ctx$terminate = TRUE)
  learner = lrn("classif.mlp", epochs = 6L, batch_size = 50, neurons = 10,
    callbacks = list(t_clbk("checkpoint", freq = 5, path = path), stopper))
  learner$train(task)

  expect_set_equal(list.files(path), c("network3.pt", "optimizer3.pt", "state3.rds"))
  expect_equal(readRDS(file.path(path, "state3.rds"))$epoch, 3L)
})

test_that("a folder that already contains checkpoints can be checkpointed into", {
  # this is what a run continuing an earlier one does: it writes epochs the folder does not have,
  # so nothing of the earlier run is touched
  task = tsk("iris")
  path = tempfile()
  first = lrn("classif.mlp", epochs = 2L, batch_size = 50, neurons = 10,
    callbacks = t_clbk("checkpoint", freq = 1, path = path))
  first$train(task)
  kept = as.numeric(torch_load(file.path(path, "network2.pt"))[[1L]]$flatten())

  # a callback pointed at that folder is accepted rather than rejected as an existing directory
  expect_no_error(t_clbk("checkpoint", freq = 1, path = path)$generate())

  # writing only epochs 3 and 4 leaves the earlier ones alone and does not warn
  writer = torch_callback("writer", weight = -1,
    on_begin = function() self$ctx$epoch = 2L)
  second = lrn("classif.mlp", epochs = 4L, batch_size = 50, neurons = 10,
    callbacks = list(writer, t_clbk("checkpoint", freq = 1, path = path)))
  expect_no_warning(second$train(task))

  expect_set_equal(list.files(path),
    c(paste0("network", 1:4, ".pt"), paste0("optimizer", 1:4, ".pt"), paste0("state", 1:4, ".rds")))
  expect_equal(as.numeric(torch_load(file.path(path, "network2.pt"))[[1L]]$flatten()), kept)
})

test_that("the checkpoint of another run is never overwritten", {
  # a run started over instead of continued used to destroy the earlier run's checkpoints
  task = tsk("iris")
  path = tempfile()
  first = lrn("classif.mlp", epochs = 1L, batch_size = 50, neurons = 10,
    callbacks = t_clbk("checkpoint", freq = 1, path = path))
  first$train(task)
  before = as.numeric(torch_load(file.path(path, "network1.pt"))[[1L]]$flatten())

  second = lrn("classif.mlp", epochs = 2L, batch_size = 50, neurons = 10,
    callbacks = t_clbk("checkpoint", freq = 1, path = path))
  expect_error(second$train(task), "were written by another run")

  # refused before the first epoch, so the earlier run's checkpoint is untouched
  after = as.numeric(torch_load(file.path(path, "network1.pt"))[[1L]]$flatten())
  expect_equal(after, before)
  expect_set_equal(list.files(path), c("network1.pt", "optimizer1.pt", "state1.rds"))
})

test_that("an incomplete leftover checkpoint may be replaced", {
  # what a run killed while writing leaves behind. It is unusable, and refusing to replace it would
  # mean such a run could never be restarted into its own folder.
  task = tsk("iris")
  path = tempfile()
  dir.create(path)
  file.create(file.path(path, c("network1.pt", "optimizer1.pt")))

  learner = lrn("classif.mlp", epochs = 1L, batch_size = 50, neurons = 10,
    callbacks = t_clbk("checkpoint", freq = 1, path = path))
  expect_no_error(learner$train(task))
  expect_set_equal(list.files(path), c("network1.pt", "optimizer1.pt", "state1.rds"))
})

test_that("latest_checkpoint() finds the most recent complete checkpoint", {
  path = tempfile()
  expect_null(latest_checkpoint(path))
  dir.create(path)
  expect_null(latest_checkpoint(path))
  expect_equal(checkpoint_suffixes(path), integer(0))

  file.create(file.path(path, c("network2.pt", "optimizer2.pt", "state2.rds",
    "network10.pt", "optimizer10.pt", "state10.rds")))
  # 10 rather than 2, i.e. the suffixes are compared as numbers and not as strings
  expect_equal(checkpoint_suffixes(path), c(10L, 2L))
  expect_equal(latest_checkpoint(path)$epoch, 10L)
  expect_equal(basename(latest_checkpoint(path)$network), "network10.pt")

  # a run that was killed while writing leaves an incomplete checkpoint, which is skipped -- but
  # not silently, as that would look like the run simply got less far than it did
  file.create(file.path(path, c("network11.pt", "optimizer11.pt")))
  expect_warning(latest_checkpoint(path), "Ignoring incomplete checkpoint\\(s\\) 11")
  expect_equal(suppressWarnings(latest_checkpoint(path))$epoch, 10L)

  # all three files are required, so the previous complete checkpoint is used instead. Without the
  # state file nothing says that the suffix counts epochs rather than within-epoch steps, which is
  # how mlr3torch <= 0.3.3 could name them.
  file.remove(file.path(path, "state10.rds"))
  expect_warning(checkpoint_suffixes(path), "checkpoint\\(s\\) 10, 11")
  expect_equal(suppressWarnings(checkpoint_suffixes(path)), 2L)
  expect_equal(suppressWarnings(latest_checkpoint(path))$epoch, 2L)
  expect_equal(basename(suppressWarnings(latest_checkpoint(path))$state), "state2.rds")

  # and a folder that holds only such checkpoints has nothing to offer
  file.remove(file.path(path, "state2.rds"))
  expect_null(suppressWarnings(latest_checkpoint(path)))
})

test_that("the saved callback states belong to the epoch of the checkpoint", {
  task = tsk("iris")
  path = tempfile()
  # the checkpoint callback is passed first, i.e. before the scheduler has stepped for the epoch
  learner = lrn("classif.mlp", epochs = 2L, batch_size = 50, neurons = 10,
    callbacks = list(t_clbk("checkpoint", freq = 1, path = path), t_clbk("lr_step")))
  learner$param_set$set_values(opt.lr = 0.1, cb.lr_step.step_size = 1, cb.lr_step.gamma = 0.5)
  learner$train(task)

  # the schedule of the last checkpoint is the one the model ends up with, not one step behind it
  state = readRDS(file.path(path, "state2.rds"))
  expect_equal(state$callbacks$lr_step$last_epoch, 2)
  expect_equal(state$callbacks$lr_step, learner$model$callbacks$lr_step)
})

test_that("the state file records the mlr3torch version", {
  path = tempfile()
  learner = lrn("classif.mlp", epochs = 2, batch_size = 50, neurons = 5,
    callbacks = t_clbk("checkpoint", path = path, freq = 1))
  learner$train(tsk("iris"))

  state = readRDS(file.path(path, "state2.rds"))
  expect_equal(state$version, as.character(packageVersion("mlr3torch")))
})

test_that("reading a checkpoint state warns on a version mismatch", {
  path = tempfile()
  learner = lrn("classif.mlp", epochs = 1, batch_size = 50, neurons = 5,
    callbacks = t_clbk("checkpoint", path = path, freq = 1))
  learner$train(tsk("iris"))
  file = file.path(path, "state1.rds")

  # the version that wrote it is the one that is running
  expect_silent({
    state = read_checkpoint_state(file)
  })
  expect_equal(state$epoch, 1L)

  written_by_other = readRDS(file)
  written_by_other$version = "0.0.1"
  saveRDS(written_by_other, file)
  expect_warning(read_checkpoint_state(file), "written by mlr3torch 0.0.1")

  # checkpoints from before the version was recorded
  written_by_old = readRDS(file)
  written_by_old$version = NULL
  saveRDS(written_by_old, file)
  expect_warning(read_checkpoint_state(file), "before checkpoints recorded one")

  # the state is returned either way, so a mismatch does not stop a resume
  expect_equal(suppressWarnings(read_checkpoint_state(file))$epoch, 1L)
})
