test_that("the early stopping state can be saved and restored", {
  task = tsk("iris")
  make = function(epochs, callbacks = list()) {
    lrn("classif.mlp", epochs = epochs, batch_size = 50, neurons = 10, validate = 0.3,
      measures_valid = msrs("classif.acc"), patience = 2L, min_delta = 0, callbacks = callbacks)
  }

  first = make(3L)
  first$train(task)
  state = first$model$callbacks$early_stopping
  expect_names(names(state), permutation.of = c("best_epochs", "best_score", "stagnation"))

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
