test_that("the early stopping state can be saved and restored", {
  task = tsk("iris")
  make = function(epochs, callbacks = list()) {
    lrn("classif.mlp", epochs = epochs, batch_size = 50, neurons = 10, validate = 0.3,
      measures_valid = msrs("classif.acc"), patience = 2L, min_delta = 0, callbacks = callbacks)
  }

  first = make(3L)
  first$train(task)
  state = first$model$callbacks$early_stopping
  expect_names(names(state), permutation.of = c("best_epochs", "best_score", "stagnation", "measure"))
  expect_equal(state$measure, "classif.acc")

  # a run that continues where the first one stopped keeps its best score and stagnation counter
  restored = NULL
  spy = torch_callback("spy",
    on_begin = function() {
      cb = self$ctx$callbacks$early_stopping
      cb$load_state_dict(state)
      restored <<- list(best_score = cb$best_score, stagnation = cb$stagnation,
        epoch_at_best_score = cb$epoch_at_best_score)
    }
  )
  second = make(3L, spy)
  second$train(task)

  expect_equal(restored$best_score, state$best_score)
  expect_equal(restored$stagnation, state$stagnation)
  expect_equal(restored$epoch_at_best_score, state$best_epochs)
})

test_that("the internally tuned epochs still come from the state dict", {
  task = tsk("iris")
  learner = lrn("classif.mlp", epochs = 3L, batch_size = 50, neurons = 10, validate = 0.3,
    measures_valid = msrs("classif.acc"), patience = 2L, min_delta = 0)
  learner$train(task)

  expect_equal(learner$internal_tuned_values$epochs,
    learner$model$callbacks$early_stopping$best_epochs)
})

test_that("restore_best_weights restores the weights of the best epoch", {
  task = tsk("mtcars")
  make_es_learner = function(...) {
    learner = lrn("regr.mlp", batch_size = 8, neurons = c(50, 50), p = 0, validate = 0.3,
      measures_valid = msr("regr.mse"), seed = 3, opt.lr = 0.5, ...)
    # the R seed decides the validation split, so every learner has to be trained under the same one
    withr::with_seed(42, learner$train(task))
    learner
  }
  state_nums = function(learner) {
    lapply(learner$model$network$state_dict(), function(x) as.numeric(x$cpu()))
  }

  expect_false(lrn("classif.mlp")$param_set$values$restore_best_weights)

  last = make_es_learner(epochs = 30, patience = 3)
  best_epoch = last$internal_tuned_values$epochs
  expect_true(best_epoch < last$model$epochs) # early stopping fired

  restored = make_es_learner(epochs = 30, patience = 3, restore_best_weights = TRUE)

  # the restored network is exactly what training for `best_epoch` epochs produces
  reference = make_es_learner(epochs = best_epoch, patience = 0)
  expect_equal(state_nums(restored), state_nums(reference))
  expect_false(isTRUE(all.equal(state_nums(last), state_nums(reference))))

  # and the flag does not perturb training itself
  expect_equal(last$model$epochs, restored$model$epochs)
  expect_equal(last$internal_tuned_values$epochs, restored$internal_tuned_values$epochs)

  # the restore runs after the checkpoint callback, so the checkpoint holds the network as training
  # left it and not the restored one
  path = tempfile()
  checkpointed = make_es_learner(epochs = 30, patience = 3, restore_best_weights = TRUE,
    callbacks = t_clbk("checkpoint", path = path, freq = 100))
  saved = lapply(torch_load(file.path(path, sprintf("network%s.pt", checkpointed$model$epochs))),
    function(x) as.numeric(x$cpu()))
  expect_equal(saved, state_nums(last))
  expect_false(isTRUE(all.equal(saved, state_nums(checkpointed))))
})

describe("resuming", {
  it("early stopping continues with the score and stagnation of the previous run", {
    # a fixed validation task and `opt.lr = 0` make the score constant, so no epoch of the resumed
    # run can beat the first run's best. Whatever the counters end up at was therefore carried over,
    # not found again -- a run starting early stopping from scratch could only name an epoch of its
    # own, i.e. 3, and would count its stagnation from zero.
    task = tsk("iris")
    task$internal_valid_task = 1:30
    path = tempfile()
    make = function(epochs, cbs) lrn("classif.mlp", epochs = epochs, batch_size = 50, neurons = 10,
      seed = 1, validate = "predefined", measures_valid = msr("classif.ce"), patience = 10L,
      min_delta = 0, opt.lr = 0, callbacks = cbs)

    make(2L, t_clbk("checkpoint", freq = 1, path = path))$train(task)
    checkpointed = readRDS(file.path(path, "state2.rds"))$callbacks$early_stopping

    resumed = make(4L, list())
    resumed$param_set$set_values(resume = path)
    resumed$train(task)

    state = resumed$model$callbacks$early_stopping
    expect_equal(checkpointed$best_epochs, 1L)
    expect_equal(state$best_epochs, checkpointed$best_epochs)
    expect_equal(state$best_score, checkpointed$best_score)
    # the two epochs of this run stagnated on top of the one the checkpoint had already counted
    expect_equal(state$stagnation, checkpointed$stagnation + 2L)
    expect_equal(resumed$model$epochs, 4L)
  })

  it("warns with restore_best_weights that the best weights are not restored", {
    path = tempfile()
    task = task_with_valid()
    args = list(validate = "predefined", measures_valid = msrs("classif.acc"), patience = 5L, min_delta = 0)
    invoke(make_checkpoint, epochs = 2L, path = path, task = task,
      .args = c(args, restore_best_weights = TRUE))

    # a full copy of the network per checkpoint is not worth it, so the weights are not stored
    expect_named(readRDS(file.path(path, "state2.rds"))$callbacks$early_stopping,
      c("best_epochs", "best_score", "stagnation", "measure"))

    resumed = invoke(resumer, epochs = 4L, path = path, .args = c(args, restore_best_weights = TRUE))
    expect_warning(resumed$train(task), "restore_best_weights")

    # without it there is nothing that could be lost, so nothing is reported
    quiet = invoke(resumer, epochs = 4L, path = path, .args = args)
    expect_no_warning(quiet$train(task))
  })

  it("refuses a checkpoint whose best score belongs to another validation measure", {
    # `best_score` is a value of one measure on one scale. Comparing this run's scores against it
    # would stop training -- or fail to -- on the difference between the two measures.
    task = task_with_valid()
    path = tempfile()
    make = function(epochs, measures, ...) lrn("classif.mlp", epochs = epochs, batch_size = 50,
      neurons = 10, seed = 1, validate = "predefined", measures_valid = measures, patience = 5L,
      min_delta = 0, ...)

    make(2L, msr("classif.acc"),
      callbacks = t_clbk("checkpoint", freq = 1, path = path))$train(task)

    other = make(4L, msr("classif.ce"), resume = path)
    expect_error(other$train(task), "value of the validation measure 'classif.acc'")

    # reordering is enough, since only the first measure is tracked
    reordered = make(4L, msrs(c("classif.ce", "classif.acc")), resume = path)
    expect_error(reordered$train(task), "but this run tracks 'classif.ce'")

    # the same first measure continues, whatever follows it
    same = make(4L, msrs(c("classif.acc", "classif.ce")), resume = path)
    expect_no_error(same$train(task))
    expect_equal(same$model$epochs, 4L)
  })

  it("a run that early stopping ended is finished", {
    # `stagnation` is restored at or above `patience`, and the loop consults `terminate` before an
    # epoch rather than after it, so such a run trains nothing at all and hands back the model of
    # the checkpoint -- otherwise a script restarting itself would creep one epoch per attempt
    task = tsk("iris")
    task$internal_valid_task = 1:30
    path = tempfile()
    make = function(...) {
      lrn("classif.mlp", epochs = 20L, batch_size = 50, neurons = 10, seed = 1,
        patience = 2L, min_delta = 0, validate = "predefined", measures_valid = msr("classif.ce"),
        opt.lr = 0, ...)
    }

    # opt.lr = 0 means the score can never improve, so the first run stops as soon as it can
    first = make(callbacks = t_clbk("checkpoint", freq = 1, path = path))
    first$train(task)
    expect_equal(first$model$epochs, 3L)
    expect_equal(first$model$callbacks$early_stopping$stagnation, 2L)
    written = list.files(path)

    resumed = make(resume = path)
    expect_warning(resumed$train(task), "already ended the run this checkpoint belongs to")
    expect_equal(resumed$model$epochs, 3L)
    expect_equal(
      as.numeric(resumed$network$parameters[[1L]]$flatten()),
      as.numeric(first$network$parameters[[1L]]$flatten())
    )
    expect_equal(resumed$internal_tuned_values$epochs, first$internal_tuned_values$epochs)
    # a run that trains nothing writes nothing
    expect_set_equal(list.files(path), written)
  })

  it("a resumed run does not train past the epoch early stopping chose", {
    # the same with a learning rate that can still improve the score: an epoch trained before the
    # loop looks at `terminate` would reset `stagnation` and let the run continue to all 20 epochs
    task = tsk("iris")
    task$internal_valid_task = seq(5, 150, by = 5)
    path = tempfile()
    make = function(...) {
      lrn("classif.mlp", epochs = 20L, batch_size = 50, neurons = c(20, 20), seed = 1,
        shuffle = FALSE, p = 0, opt.lr = 0.05, predict_type = "prob", patience = 2L,
        min_delta = 0.005, validate = "predefined", measures_valid = msr("classif.logloss"), ...)
    }

    first = make(callbacks = t_clbk("checkpoint", freq = 1, path = path))
    first$train(task)
    expect_lt(first$model$epochs, 20L)

    resumed = make(resume = path)
    expect_warning(resumed$train(task), "already ended the run this checkpoint belongs to")
    expect_equal(resumed$model$epochs, first$model$epochs)
    expect_equal(resumed$internal_tuned_values$epochs, first$internal_tuned_values$epochs)
  })
})
