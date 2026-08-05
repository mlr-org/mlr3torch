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
