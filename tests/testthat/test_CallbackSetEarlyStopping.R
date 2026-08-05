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
    permutation.of = c("best_epochs", "best_score", "best_valid_scores", "stagnation"))

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

test_that("best_valid_scores are the scores of the best epoch", {
  task = tsk("iris")
  learner = lrn("classif.mlp", epochs = 5L, batch_size = 50, neurons = 10, validate = 0.3,
    measures_valid = msrs(c("classif.acc", "classif.ce")), patience = 2L, min_delta = 0)
  learner$train(task)

  best = learner$best_valid_scores
  last = learner$internal_valid_scores
  expect_list(best, types = "numeric")
  # all validation measures are tracked, not just the one early stopping looks at
  expect_names(names(best), permutation.of = names(last))
  # the first measure is the one early stopping optimizes, so it holds the best score
  expect_equal(best[[1L]], learner$model$callbacks$early_stopping$best_score)
  # the best epoch is the one reported as the internally tuned value
  expect_equal(learner$internal_tuned_values$epochs,
    learner$model$callbacks$early_stopping$best_epochs)
  # classif.acc is maximized, so the best epoch is at least as good as the last one
  expect_true(best$classif.acc >= last$classif.acc)

  # msr("best_valid_score") reads them
  rr = resample(task, learner, rsmp("holdout"))
  expect_equal(
    rr$score(msr("best_valid_score", select = "classif.acc"))$classif.acc,
    rr$learners[[1]]$best_valid_scores$classif.acc
  )
})

test_that("no best_valid_scores without early stopping", {
  task = tsk("iris")
  learner = lrn("classif.mlp", epochs = 2L, batch_size = 50, neurons = 10, validate = 0.3,
    measures_valid = msrs("classif.acc"), patience = 0L)
  learner$train(task)

  expect_equal(learner$best_valid_scores, named_list())
  expect_list(learner$internal_valid_scores, types = "numeric")
})

test_that("the internally tuned epochs still come from the state dict", {
  task = tsk("iris")
  learner = lrn("classif.mlp", epochs = 3L, batch_size = 50, neurons = 10, validate = 0.3,
    measures_valid = msrs("classif.acc"), patience = 2L, min_delta = 0)
  learner$train(task)

  expect_equal(learner$internal_tuned_values$epochs,
    learner$model$callbacks$early_stopping$best_epochs)
})
