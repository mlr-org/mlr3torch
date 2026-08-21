test_that("a measure can read the task", {
  d = tt_data(40L)
  d$y = rnorm(nrow(d))
  task = tt_task(d, target = "y")
  measure = msr_torch("n", function(task, prediction) length(prediction$row_ids) / task$nrow)

  expect_true("requires_task" %chin% measure$properties)
  learner = tt_learner(t_loss("mse"))
  learner$train(task)
  expect_equal(learner$predict(task)$score(measure, task = task), c(n = 1))
})

test_that("a measure declares what it asks for", {
  expect_equal(msr_torch("a", function(truth, response) 1)$properties, character(0))
  expect_set_equal(
    msr_torch("b", function(task, learner, train_set, response) 1)$properties,
    c("requires_task", "requires_learner", "requires_train_set")
  )

  # the `obs_loss` is scored through the same arguments, so it declares them too
  expect_set_equal(
    msr_torch("c", function(truth) 1, obs_loss = function(truth, learner, weights) 1)$properties,
    c("obs_loss", "requires_learner", "weights")
  )

  # what the caller declares is added to what is derived, rather than replaced -- a measure reading
  # `learner$network` is meant to be written exactly like this, see `?msr_torch`
  expect_set_equal(
    msr_torch("d", function(truth, learner) 1, properties = "requires_model")$properties,
    c("requires_learner", "requires_model")
  )

  # the arguments actually arrive, which resample() is what provides them
  d = tt_data(40L)
  d$y = rnorm(nrow(d))
  task = tt_task(d, target = "y")
  measure = msr_torch("seen", function(learner, train_set) {
    stopifnot(inherits(learner, "LearnerTorch"))
    length(train_set)
  })
  rr = resample(task, tt_learner(t_loss("mse")), rsmp("holdout", ratio = 0.5))
  expect_equal(unname(rr$aggregate(measure)), 20)
})

test_that("an argument the prediction does not have keeps its default", {
  d = tt_data(40L)
  d$y = rnorm(nrow(d))
  task = tt_task(d, target = "y")
  learner = tt_learner(t_loss("mse"))
  learner$train(task)
  pred = learner$predict(task)

  # the task has no `weights_measure` column, so `weights` is NULL here; passing it on explicitly
  # would override the default and score `sum(... * NULL) / n`, i.e. a perfect 0
  expect_null(pred$weights)
  weighted = msr_torch("wmse", function(truth, response, weights = rep(1, length(truth))) {
    sum((truth - response)^2 * weights) / length(truth)
  }, range = c(0, Inf))
  expect_equal(unname(pred$score(weighted)), mean((pred$truth - pred$response)^2))

  # ... while an argument without a default still arrives, as NULL
  expect_equal(unname(pred$score(msr_torch("se_null", function(truth, se) 1 * is.null(se)))), 1)
})

test_that("a measure can report a per observation loss", {
  d = tt_data(40L)
  d$y = rnorm(nrow(d))
  task = tt_task(d, target = "y")

  measure = msr_torch("mse", function(truth, response) mean((truth - response)^2),
    obs_loss = function(truth, response) (truth - response)^2, range = c(0, Inf))
  expect_true("obs_loss" %chin% measure$properties)

  learner = tt_learner(t_loss("mse"))
  learner$train(task)
  pred = learner$predict(task)
  losses = measure$obs_loss(pred)
  expect_numeric(losses, len = task$nrow, lower = 0, any.missing = FALSE)
  expect_equal(mean(losses), unname(pred$score(measure)))

  # `Measure` reports NA without the property, rather than erroring on the missing function
  plain = msr_torch("plain", function(truth, response) mean((truth - response)^2))
  expect_true("obs_loss" %nin% plain$properties)
  expect_true(all(is.na(plain$obs_loss(pred))))

  # ... and it reaches the place that reports it
  rr = resample(task, learner, rsmp("cv", folds = 2L))
  tab = rr$obs_loss(measure)
  expect_data_table(tab, nrows = task$nrow)
  expect_numeric(tab$mse, lower = 0, any.missing = FALSE)
})

test_that("an obs_loss has to return one value per observation", {
  d = tt_data(40L)
  d$y1 = rnorm(nrow(d))
  d$y2 = rnorm(nrow(d))
  task = tt_task(d, target = c("y1", "y2"))
  learner = tt_learner(t_loss("mse"))
  learner$train(task)
  pred = learner$predict(task)

  # `mean()` where `rowMeans()` was meant: `data.table::set()` would recycle the one number over
  # every observation, which looks exactly like a per-observation loss
  reduced = msr_torch("reduced", function(truth, response) 1,
    obs_loss = function(truth, response) mean((as.matrix(truth) - response)^2))
  expect_error(pred$obs_loss(reduced), "returned 1 values for 40 observations")

  # ... and it is not a number at all
  wrong_type = msr_torch("chr", function(truth, response) 1,
    obs_loss = function(truth, response) rep("a", nrow(response)))
  expect_error(pred$obs_loss(wrong_type), "value of the `obs_loss` of measure 'chr'")

  ok = msr_torch("rowwise", function(truth, response) 1,
    obs_loss = function(truth, response) rowMeans((as.matrix(truth) - response)^2))
  expect_numeric(pred$obs_loss(ok)$rowwise, len = task$nrow, any.missing = FALSE)
})

test_that("the default measure does not fix an optimization direction", {
  expect_true(is.na(msr("torch.default")$minimize))
})

test_that("a measure rejects arguments it could never be given", {
  # the typo would otherwise surface at score time as `argument "respones" is missing`, and inside
  # a training run as `Measure 'x' could not be computed and is reported as NaN`
  expect_error(msr_torch("t", function(truth, respones) 1), "arguments of `fun`")
  expect_error(msr_torch("t", function(truth, response) 1, obs_loss = function(truth, respones) 1),
    "arguments of `obs_loss`")
  # `train_set` is one of the supported arguments, but `Measure$obs_loss()` is not given one
  expect_error(msr_torch("t", function(truth) 1, obs_loss = function(truth, train_set) 1),
    "arguments of `obs_loss`")
})

test_that("a measure can read the predicted tensors", {
  d = tt_data(20L)
  d$y = rnorm(nrow(d))
  task = tt_task(d, target = "y")
  learner = tt_learner(t_loss("mse"), predict_types = c("response", "lazy_tensor"))
  learner$predict_type = "lazy_tensor"
  learner$train(task)
  pred = learner$predict(task)

  measure = msr_torch("norm", function(lazy_tensor) {
    as.numeric(materialize(lazy_tensor, rbind = TRUE)$pow(2)$mean()$cpu())
  }, predict_type = "lazy_tensor", range = c(0, Inf))
  expect_number(pred$score(measure), lower = 0)
})

test_that("the default measure delegates the per observation loss", {
  d = tt_data(20L)
  d$y = rnorm(nrow(d))
  learner = tt_learner(t_loss("mse"))
  default = msr("torch.default")

  with_obs_loss = tt_task(d, target = "y", id = "t", default_measure = msr_torch("mse",
    function(truth, response) mean((truth - response)^2),
    obs_loss = function(truth, response) (truth - response)^2, range = c(0, Inf)))
  learner$train(with_obs_loss)
  pred = learner$predict(with_obs_loss)
  expect_equal(
    default$obs_loss(pred, task = with_obs_loss),
    (pred$truth - pred$response)^2
  )

  # a task whose measure has none behaves like any other measure without one, i.e. reports NA
  without = tt_task(d, target = "y", id = "t",
    default_measure = msr_torch("mse", function(truth, response) mean((truth - response)^2)))
  expect_true(all(is.na(default$obs_loss(learner$predict(without), task = without))))
})

test_that("a measure can be scored with observation weights", {
  d = tt_data(40L)
  d$a = d$x1 > 0
  d$b = d$x2 > 0
  # only the second half counts, so a weighted score must differ from the unweighted one
  d$w = c(rep(0, 20L), rep(2, 20L))
  task = tt_task(d, target = c("a", "b"), id = "w")
  task$set_col_roles("w", "weights_measure")

  learner = tt_learner(tt_loss_bce())
  learner$train(task)
  pred = learner$predict(task)
  expect_numeric(pred$data$weights, len = task$nrow)

  # declaring a `weights` argument adds the property, which is what lets mlr3 score at all
  weighted = msr_torch("wham",
    function(truth, response, weights) weighted.mean(rowMeans(as.matrix(truth) != response), weights),
    range = c(0, 1))
  expect_true("weights" %chin% weighted$properties)

  # the values that arrive are the task's column, not something made up
  expect_equal(unname(pred$score(msr_torch("sum_w", function(weights) sum(weights),
    range = c(0, Inf)), task = task)), sum(d$w))

  # ... and they reach the scoring function aligned with the observations: weighting the first
  # half out has to give exactly the unweighted score over the second half
  errs = rowMeans(as.matrix(pred$truth) != pred$response)
  expect_equal(unname(pred$score(weighted, task = task)), mean(errs[pred$row_ids > 20L]))

  # the alignment is by row, not by position: a prediction in a different row order scores the same
  shuffled = learner$predict(task, row_ids = rev(task$row_ids))
  expect_equal(unname(shuffled$score(weighted, task = task)),
    unname(pred$score(weighted, task = task)))
})

test_that("a measure respects use_weights", {
  d = tt_data(40L)
  d$a = d$x1 > 0
  d$b = d$x2 > 0
  d$w = c(rep(0, 20L), rep(2, 20L))
  task = tt_task(d, target = c("a", "b"), id = "w")
  task$set_col_roles("w", "weights_measure")

  learner = tt_learner(tt_loss_bce())
  learner$train(task)
  pred = learner$predict(task)
  errs = rowMeans(as.matrix(pred$truth) != pred$response)

  # `mlr3` hands over the weights or `NULL`, depending on this switch, so a measure taking them
  # from the prediction itself would score the weighted number either way
  weighted = msr_torch("wham", function(truth, response, weights = rep(1, length(errs))) {
    weighted.mean(rowMeans(as.matrix(truth) != response), weights)
  }, range = c(0, 1), obs_loss = function(truth, response, weights = rep(1, NROW(truth))) {
    weights * rowMeans(as.matrix(truth) != response)
  })
  expect_equal(weighted$use_weights, "use")
  expect_equal(unname(pred$score(weighted, task = task)), mean(errs[pred$row_ids > 20L]))
  expect_equal(pred$obs_loss(weighted)$wham, d$w[pred$row_ids] * errs)

  weighted$use_weights = "ignore"
  expect_equal(unname(pred$score(weighted, task = task)), mean(errs))
  expect_equal(pred$obs_loss(weighted)$wham, errs)

  # ... and a function that cannot do without them says so, rather than scoring `sum(x * NULL)`
  strict = msr_torch("strict", function(truth, response, weights) sum(weights), range = c(0, Inf))
  strict$use_weights = "ignore"
  expect_error(pred$score(strict, task = task), "asks for `weights` without a default")

  # the same for a task that has no weights to begin with, which the trained network can predict
  # as well -- the weights column was never a feature
  strict$use_weights = "use"
  unweighted = tt_task(d[, !"w"], target = c("a", "b"), id = "unweighted")
  expect_error(learner$predict(unweighted)$score(strict, task = unweighted),
    "asks for `weights` without a default")
})

test_that("the default measure of a task is used by aggregate()", {
  d = tt_data(40L)
  d$y = rnorm(nrow(d))
  measure = msr_torch("mse", function(truth, response) mean((truth - response)^2), range = c(0, Inf))

  without = tt_task(d, target = "y", id = "t")
  rr = resample(without, tt_learner(t_loss("mse")), rsmp("holdout"))
  expect_error(rr$aggregate(), "has no default measure")

  with = tt_task(d, target = "y", id = "t", default_measure = measure)
  rr = resample(with, tt_learner(t_loss("mse")), rsmp("holdout"))
  expect_number(rr$aggregate(), lower = 0)
})

test_that("the hash of a measure covers its scoring function", {
  # Measure$hash covers the private .score() method, which is identical for every MeasureTorch,
  # so without an override two measures differing only in their function would collide -- and
  # since a task carries its measure, so would two tasks.
  expect_false(msr_torch("m", function(truth, response) 1)$hash ==
    msr_torch("m", function(truth, response) 2)$hash)
  expect_equal(msr_torch("m", function(truth, response) 1)$hash,
    msr_torch("m", function(truth, response) 1)$hash)

  d = tt_data(40L)
  d$y = rnorm(nrow(d))
  task = function(measure) tt_task(d, target = "y", id = "same", default_measure = measure)
  expect_false(task(msr_torch("m", function(truth, response) 1))$hash ==
    task(msr_torch("m", function(truth, response) 2))$hash)

  # ... which is what stops benchmark() from scoring both rows with one task's measure
  bmr = benchmark(benchmark_grid(
    list(task(msr_torch("m", function(truth, response) 1)),
      task(msr_torch("m", function(truth, response) 999))),
    tt_learner(t_loss("mse")), rsmp("holdout")))
  expect_equal(bmr$aggregate(msr("torch.default"))$torch.default, c(1, 999))
})
