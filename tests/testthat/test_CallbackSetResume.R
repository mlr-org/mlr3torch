checkpoint_learner = function(epochs = 2L, path, ...) {
  learner = lrn("classif.mlp", epochs = epochs, batch_size = 50, neurons = 10,
    callbacks = list(t_clbk("checkpoint", freq = 1), t_clbk("history")),
    measures_train = msrs("classif.acc"), ...)
  learner$param_set$set_values(cb.checkpoint.path = path)
  learner
}

resume_learner = function(epochs, ..., callbacks = list(t_clbk("resume"), t_clbk("history"))) {
  learner = lrn("classif.mlp", epochs = epochs, batch_size = 50, neurons = 10,
    callbacks = callbacks, measures_train = msrs("classif.acc"))
  learner$param_set$set_values(...)
  learner
}

test_that("Autotest", {
  path = tempfile()
  dir.create(path)
  cb = t_clbk("resume", path = path)
  expect_torch_callback(cb)
})

test_that("resuming continues from the checkpointed epoch", {
  task = tsk("iris")
  path = tempfile()
  checkpoint_learner(epochs = 2L, path = path)$train(task)

  learner = resume_learner(epochs = 5L, cb.resume.path = path)
  learner$train(task)

  # 'epochs' is the total number of epochs, so 3 more epochs are trained
  expect_equal(learner$model$epochs, 5L)
})

test_that("resuming loads the checkpointed weights", {
  task = tsk("iris")
  path = tempfile()
  checkpoint_learner(epochs = 2L, path = path)$train(task)
  saved = torch_load(file.path(path, "network2.pt"))

  # nothing left to train, so the weights must be exactly those of the checkpoint
  learner = resume_learner(epochs = 2L, cb.resume.path = path)
  expect_warning(learner$train(task), "No further epochs are trained")
  expect_equal(learner$model$epochs, 2L)

  weights = learner$model$network$state_dict()
  expect_true(all(pmap_lgl(list(weights, saved[names(weights)]), function(x, y) {
    as.logical(torch_equal(x, y))
  })))
})

test_that("the history of the previous run is continued", {
  task = tsk("iris")
  path = tempfile()
  checkpoint_learner(epochs = 2L, path = path)$train(task)

  learner = resume_learner(epochs = 4L, cb.resume.path = path)
  learner$train(task)

  history = learner$model$callbacks$history[order(get("epoch"))]
  expect_data_table(history, nrows = 4L)
  expect_equal(history$epoch, 1:4)
})

test_that("callbacks that cannot restore their state are skipped with a warning", {
  task = tsk("iris")
  path = tempfile()
  # a callback that has a state but cannot load it back
  cb_broken = torch_callback("Broken",
    on_epoch_end = function() self$counter = self$ctx$epoch,
    state_dict = function() list(counter = self$counter),
    load_state_dict = function(state_dict) NULL
  )
  learner = lrn("classif.mlp", epochs = 1L, batch_size = 50, neurons = 10,
    callbacks = list(t_clbk("checkpoint", freq = 1), cb_broken))
  learner$param_set$set_values(cb.checkpoint.path = path)
  learner$train(task)

  # the state is in the checkpoint, but the resuming run has no callback of that id
  learner = resume_learner(epochs = 2L, cb.resume.path = path)
  expect_warning(learner$train(task), "not part of this training run")
  expect_equal(learner$model$epochs, 2L)
})

test_that("training starts from scratch when there is no checkpoint yet", {
  task = tsk("iris")
  # the same script can be used for the first run and for restarts
  learner = resume_learner(epochs = 2L, cb.resume.path = tempfile())
  learner$train(task)
  expect_equal(learner$model$epochs, 2L)
  expect_data_table(learner$model$callbacks$history, nrows = 2L)
})

test_that("explicit network and optimizer paths are used", {
  task = tsk("iris")
  path = tempfile()
  checkpoint_learner(epochs = 2L, path = path)$train(task)
  saved = torch_load(file.path(path, "network1.pt"))

  # such a checkpoint carries no epoch information, so all epochs are trained
  learner = resume_learner(epochs = 3L,
    cb.resume.network_path = file.path(path, "network1.pt"),
    cb.resume.optimizer_path = file.path(path, "optimizer1.pt"))
  learner$train(task)
  expect_equal(learner$model$epochs, 3L)
  expect_data_table(learner$model$callbacks$history, nrows = 3L)

  # ... unless epochs_trained says otherwise
  learner = resume_learner(epochs = 3L,
    cb.resume.network_path = file.path(path, "network1.pt"),
    cb.resume.optimizer_path = file.path(path, "optimizer1.pt"),
    cb.resume.epochs_trained = 1L)
  learner$train(task)
  expect_equal(learner$model$epochs, 3L)
  expect_data_table(learner$model$callbacks$history, nrows = 2L)
})

test_that("checkpoints without a state file are resumed via the file suffix", {
  task = tsk("iris")
  path = tempfile()
  checkpoint_learner(epochs = 2L, path = path)$train(task)
  # a checkpoint as written by earlier versions of mlr3torch
  file.remove(list.files(path, pattern = "^state", full.names = TRUE))

  learner = resume_learner(epochs = 3L, cb.resume.path = path)
  learner$train(task)
  expect_equal(learner$model$epochs, 3L)
  # the history of the previous run is gone, only the new epoch is recorded
  expect_data_table(learner$model$callbacks$history, nrows = 1L)
  expect_equal(learner$model$callbacks$history$epoch, 3L)
})

test_that("the most recent checkpoint is used", {
  task = tsk("iris")
  path = tempfile()
  checkpoint_learner(epochs = 3L, path = path)$train(task)
  # an incomplete checkpoint (e.g. the job died while writing) must not be picked
  file.remove(file.path(path, "optimizer3.pt"))

  learner = resume_learner(epochs = 4L, cb.resume.path = path)
  learner$train(task)
  expect_equal(learner$model$epochs, 4L)
  # resumed from epoch 2, so epochs 3 and 4 were trained
  expect_data_table(learner$model$callbacks$history, nrows = 4L)
})

test_that("input is checked", {
  expect_error(CallbackSetResume$new(), "must be provided")
  expect_error(CallbackSetResume$new(network_path = tempfile()), "must either both be provided")
  expect_error(CallbackSetResume$new(network_path = tempfile(), optimizer_path = tempfile()),
    "does not exist")
})

test_that("checkpointing into the folder that is resumed from works", {
  task = tsk("iris")
  path = tempfile()
  checkpoint_learner(epochs = 2L, path = path)$train(task)

  # this is the restart-the-same-script workflow: resume and keep checkpointing
  learner = resume_learner(epochs = 4L,
    callbacks = list(t_clbk("resume"), t_clbk("checkpoint", freq = 1), t_clbk("history")),
    cb.resume.path = path, cb.checkpoint.path = path)
  learner$train(task)

  expect_equal(learner$model$epochs, 4L)
  expect_true(all(file.exists(file.path(path, paste0("network", 1:4, ".pt")))))
  expect_equal(readRDS(file.path(path, "state4.rds"))$epoch, 4L)
})

test_that("early stopping continues with the score and stagnation of the previous run", {
  task = tsk("iris")
  path = tempfile()
  make = function(epochs, callbacks) {
    lrn("classif.mlp", epochs = epochs, batch_size = 50, neurons = 10, patience = 5L,
      validate = 0.3, measures_valid = msrs("classif.acc"), callbacks = callbacks)
  }

  learner = make(3L, t_clbk("checkpoint", freq = 1))
  learner$param_set$set_values(cb.checkpoint.path = path)
  learner$train(task)

  state = readRDS(file.path(path, "state3.rds"))$callbacks$early_stopping
  expect_names(names(state), permutation.of = c("best_epochs", "best_score", "stagnation"))

  resumed = make(5L, t_clbk("resume"))
  resumed$param_set$set_values(cb.resume.path = path)
  resumed$train(task)

  # the best epoch of the first run is still known, so it can be reported as the tuned value
  expect_equal(resumed$internal_tuned_values$epochs, resumed$model$callbacks$early_stopping$best_epochs)
  expect_false(is.null(resumed$model$callbacks$early_stopping$best_score))
})

test_that("stateless callbacks do not interfere with resuming", {
  task = tsk("iris")
  path = tempfile()
  learner = lrn("classif.mlp", epochs = 2L, batch_size = 50, neurons = 10,
    callbacks = list(t_clbk("progress"), t_clbk("checkpoint", freq = 1)))
  learner$param_set$set_values(cb.checkpoint.path = path)
  capture.output(learner$train(task))

  expect_null(readRDS(file.path(path, "state2.rds"))$callbacks$progress)

  resumed = lrn("classif.mlp", epochs = 3L, batch_size = 50, neurons = 10,
    callbacks = list(t_clbk("resume"), t_clbk("progress")))
  resumed$param_set$set_values(cb.resume.path = path)
  capture.output(resumed$train(task))
  expect_equal(resumed$model$epochs, 3L)
})
