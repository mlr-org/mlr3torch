# A `PredictionTorch` does not prescribe how its elements are stored, so what `mlr3` does with a
# prediction has to keep working for whatever the task's `prediction_encoder` produced. The tests
# below run the operations that `mlr3` performs on any prediction, see `?mlr3::Prediction`.

# a regression-shaped task and a trained learner, the plainest thing a `TaskTorch` can be
tt_fitted = function(n = 40L) {
  d = tt_data(n)
  d$y = d$x1 + rnorm(nrow(d))
  task = tt_task(d, target = "y")
  learner = tt_learner(t_loss("mse"))
  learner$train(task)
  list(task = task, learner = learner, prediction = learner$predict(task))
}

test_that("a prediction can be constructed from a task", {
  d = tt_data(10L)
  d$y = rnorm(nrow(d))
  task = tt_task(d, target = "y")

  pred = PredictionTorch$new(task, response = rnorm(task$nrow))
  expect_class(pred, c("PredictionTorch", "Prediction"))
  expect_equal(pred$task_type, "torch")
  expect_equal(pred$row_ids, task$row_ids)
  expect_equal(pred$truth, task$truth())
  expect_equal(pred$predict_types, "response")
  expect_null(pred$prob)
  expect_null(pred$se)

  # the row ids and the elements have to describe the same observations
  expect_error(PredictionTorch$new(task, response = rnorm(task$nrow - 1L)),
    "has 9 observations, but 10 row ids")
})

test_that("a prediction can be scored", {
  fitted = tt_fitted()
  measure = msr_torch("mse", function(truth, response) mean((truth - response)^2),
    range = c(0, Inf))

  score = fitted$prediction$score(measure)
  expect_number(score, lower = 0)
  expect_named(score, "mse")

  # the id of the measure, not of the task, is what names the score
  expect_equal(unname(score), mean((fitted$prediction$truth - fitted$prediction$response)^2))
})

test_that("a prediction becomes a data.table", {
  fitted = tt_fitted()
  tab = as.data.table(fitted$prediction)

  expect_data_table(tab, nrows = fitted$task$nrow)
  expect_set_equal(names(tab), c("row_ids", "truth", "response"))
  expect_equal(tab$row_ids, fitted$prediction$row_ids)
  expect_equal(tab$truth, fitted$prediction$truth)
})

test_that("a prediction can be printed", {
  fitted = tt_fitted()
  # `Prediction$print()` goes through `as.data.table()`, so this exercises the method above
  expect_output(print(fitted$prediction), "PredictionTorch")
  expect_output(print(fitted$prediction), "40")

  empty = fitted$learner$predict(fitted$task, row_ids = integer(0))
  expect_output(print(empty), "0")
})

test_that("a prediction can be filtered", {
  fitted = tt_fitted()
  pred = fitted$prediction
  keep = pred$row_ids[c(2L, 5L, 7L)]
  before = pred$response[c(2L, 5L, 7L)]

  pred$filter(keep)
  expect_equal(pred$row_ids, keep)
  expect_equal(pred$response, before)
  expect_length(pred$truth, 3L)
  expect_data_table(as.data.table(pred), nrows = 3L)

  # filtering to row ids that are not in the prediction leaves nothing
  pred$filter(max(fitted$task$row_ids) + 1L)
  expect_length(pred$row_ids, 0L)
})

test_that("two predictions can be combined", {
  fitted = tt_fitted()
  rows = fitted$task$row_ids
  a = fitted$learner$predict(fitted$task, row_ids = rows[1:10])
  b = fitted$learner$predict(fitted$task, row_ids = rows[11:40])

  combined = c(a, b)
  expect_class(combined, "PredictionTorch")
  expect_equal(combined$row_ids, rows)
  expect_equal(combined$predict_types, "response")
  expect_equal(combined$truth, fitted$task$truth(rows))
  expect_data_table(as.data.table(combined), nrows = 40L)

  # the same observation twice, unless it is asked to drop duplicates
  expect_length(c(a, a)$row_ids, 20L)
  expect_length(c(a, a, keep_duplicates = FALSE)$row_ids, 10L)
})

test_that("a prediction survives a round trip through its prediction data", {
  fitted = tt_fitted()
  pdata = as_prediction_data(fitted$prediction, task = fitted$task)
  expect_class(pdata, "PredictionDataTorch")

  back = as_prediction(pdata)
  expect_class(back, "PredictionTorch")
  expect_equal(back$row_ids, fitted$prediction$row_ids)
  expect_equal(back$response, fitted$prediction$response)
  expect_equal(back$truth, fitted$prediction$truth)
})

test_that("missing predictions are reported", {
  d = tt_data(10L)
  d$y = rnorm(nrow(d))
  task = tt_task(d, target = "y")

  response = rnorm(task$nrow)
  response[c(3L, 8L)] = NA_real_
  pred = PredictionTorch$new(task, response = response)
  expect_equal(pred$missing, task$row_ids[c(3L, 8L)])

  expect_length(PredictionTorch$new(task, response = rnorm(task$nrow))$missing, 0L)
})

test_that("probabilities and standard errors are carried along", {
  d = tt_data(10L)
  d$y = rnorm(nrow(d))
  task = tt_task(d, target = "y")
  n = task$nrow

  pred = PredictionTorch$new(task, response = rnorm(n), prob = matrix(runif(2L * n), ncol = 2L),
    se = runif(n))
  expect_set_equal(pred$predict_types, c("response", "prob", "se"))
  expect_matrix(pred$prob, nrows = n, ncols = 2L)
  expect_numeric(pred$se, len = n)

  # every element is filtered, not just the response
  pred$filter(task$row_ids[1:4])
  expect_matrix(pred$prob, nrows = 4L, ncols = 2L)
  expect_numeric(pred$se, len = 4L)

  # and they reach the table, the `prob` matrix as one column per column
  tab = as.data.table(pred)
  expect_data_table(tab, nrows = 4L)
  expect_true("se" %chin% names(tab))
  expect_equal(sum(startsWith(names(tab), "prob")), 2L)
})

test_that("predictions with different predict types do not combine", {
  d = tt_data(10L)
  d$y = rnorm(nrow(d))
  task = tt_task(d, target = "y")

  a = PredictionTorch$new(task, response = rnorm(task$nrow))
  b = PredictionTorch$new(task, response = rnorm(task$nrow), se = runif(task$nrow))
  # `mlr3::c.Prediction()` does not check, so silently dropping the `se` would be the alternative
  expect_error(c(a, b), "different predict types")
})

test_that("the fields of a prediction are read only", {
  fitted = tt_fitted()
  expect_error(fitted$prediction$response <- 1, "read-only")
  expect_error(fitted$prediction$prob <- 1, "read-only")
  expect_error(fitted$prediction$se <- 1, "read-only")
})
