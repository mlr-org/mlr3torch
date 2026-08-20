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

test_that("the default measure does not fix an optimization direction", {
  expect_true(is.na(msr("torch.default")$minimize))
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
