test_that("the early stopping state can be saved and restored", {
  task = tsk("iris")
  make = function(epochs, callbacks = list()) {
    lrn("classif.mlp", epochs = epochs, batch_size = 50, neurons = 10, validate = 0.3,
      measures_valid = msrs("classif.acc"), patience = 2L, min_delta = 0, callbacks = callbacks)
  }

  first = make(3L)
  first$train(task)
  state = first$model$callbacks$early_stopping
  expect_names(names(state),
    permutation.of = c("best_epochs", "best_score", "best_scores", "stagnation"))

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

test_that("internal_valid_scores and internal_tuned_values describe the same epoch", {
  # with early stopping the stored network is the last epoch's, but both extractors must describe
  # the *best* epoch, because mlr3tuning pairs them as if they came from one model
  task = tsk("mtcars")
  learner = lrn("regr.mlp", epochs = 30, batch_size = 8, neurons = c(200, 200), p = 0,
    patience = 3, validate = 0.3, measures_valid = msr("regr.mse"), seed = 3, opt.lr = 0.5,
    callbacks = t_clbk("history"))
  learner$train(task)

  history = learner$model$callbacks$history
  best_epoch = learner$internal_tuned_values$epochs
  expect_int(best_epoch)
  expect_true(best_epoch < learner$model$epochs) # early stopping actually fired

  expect_equal(
    learner$internal_valid_scores$regr.mse,
    history[get("epoch") == best_epoch, ][["valid.regr.mse"]]
  )
  # and it is genuinely the best score, not merely the last one
  expect_equal(learner$internal_valid_scores$regr.mse, min(history$valid.regr.mse))
})

test_that("internal_valid_scores reports every measure at the best epoch", {
  task = tsk("iris")
  learner = lrn("classif.mlp", epochs = 10L, batch_size = 50, neurons = 10, validate = 0.3,
    measures_valid = msrs(c("classif.acc", "classif.ce")), patience = 2L, seed = 1)
  learner$train(task)

  expect_names(names(learner$internal_valid_scores), permutation.of = c("classif.acc", "classif.ce"))
  # the two measures are complementary, so this pins that both come from the same epoch
  expect_equal(learner$internal_valid_scores$classif.acc,
    1 - learner$internal_valid_scores$classif.ce)
})

test_that("without early stopping the scores are still the last epoch's", {
  task = tsk("iris")
  learner = lrn("classif.mlp", epochs = 3L, batch_size = 50, neurons = 10, validate = 0.3,
    measures_valid = msrs("classif.acc"), patience = 0L, callbacks = t_clbk("history"))
  learner$train(task)

  expect_equal(learner$internal_tuned_values, named_list())
  expect_equal(learner$internal_valid_scores$classif.acc,
    learner$model$callbacks$history[get("epoch") == 3L, ][["valid.classif.acc"]])
})
