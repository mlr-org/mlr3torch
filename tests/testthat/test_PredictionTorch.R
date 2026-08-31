# A `PredictionTorch` does not prescribe how its elements are stored, so what `mlr3` does with a
# prediction has to keep working for whatever the task's `default_encoder` produced. The tests
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

  # and they reach the table, the `prob` matrix as one column per class -- which is what `mlr3`
  # does for a classification prediction, and the one element that is tabled that way
  tab = as.data.table(pred)
  expect_data_table(tab, nrows = 4L)
  expect_true("se" %chin% names(tab))
  expect_equal(sum(startsWith(names(tab), "prob")), 2L)
  expect_false(inherits(tab$prob.V1, "pt_arrays"))
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

test_that("combining prediction data with different predict types errors", {
  d = tt_data(6L)
  d$a = d$x1 > 0
  task = tt_task(d, target = "a", id = "t")

  without = PredictionTorch$new(task, row_ids = 1:3, truth = task$truth(1:3),
    response = c(TRUE, FALSE, TRUE))$data
  with = PredictionTorch$new(task, row_ids = 4:6, truth = task$truth(4:6),
    response = c(TRUE, FALSE, TRUE), prob = c(0.9, 0.1, 0.8))$data

  # would previously have dropped `prob` without a word
  expect_error(c(without, with), "different predict types")
  expect_error(c(with, without), "different predict types")

  expect_equal(length(c(without, without)$row_ids), 6L)
  expect_equal(length(c(with, with)$prob), 6L)
  expect_equal(length(c(without)$row_ids), 3L)
})

test_that("predicting on zero rows gives an empty prediction", {
  task = tt_task_labels(40L, id = "t")
  learner = tt_learner(tt_loss_bce(), predict_types = c("response", "prob"))
  learner$predict_type = "prob"
  learner$train(task)

  pred = learner$predict(task$clone(deep = TRUE)$filter(integer(0)))
  expect_class(pred, "PredictionTorch")
  expect_equal(length(pred$row_ids), 0L)
  # the prediction still says what it is: degrading to row ids and truth alone only shows up much
  # later, as a `different predict types` error when it is combined with a real one
  expect_set_equal(pred$predict_types, c("response", "prob"))
  expect_matrix(pred$response, nrows = 0L, ncols = 2L)

  # an empty prediction must have the same storage as a non-empty one, so the two can be combined
  empty = create_empty_prediction_data(task, learner)
  expect_names(names(empty), permutation.of = c("row_ids", "truth", "response", "prob"))
  expect_matrix(empty$response, nrows = 0L, ncols = 2L)

  combined = c(empty, learner$predict(task)$data)
  expect_matrix(combined$response, nrows = task$nrow, ncols = 2L)
  expect_matrix(combined$prob, nrows = task$nrow, ncols = 2L)
  expect_equal(combined$row_ids, task$row_ids)
})

test_that("an encoder that cannot build an empty prediction errors here", {
  # degrading to row ids and truth alone would only surface much later, as a `different predict
  # types` error the moment the empty prediction meets a real one
  d = tt_data(20L)
  d$y = rnorm(nrow(d))
  task = tt_task(d, target = "y", id = "e",
    default_encoder = function(task, network_output, predict_type) {
      if (!nrow(as.matrix(network_output$cpu()))) stopf("no empty batches here")
      list(response = as.numeric(as.matrix(network_output$cpu())))
    })
  learner = tt_learner(t_loss("mse"))
  learner$train(task)

  expect_error(create_empty_prediction_data(task, learner), "no empty batches here")

  # ... but a task that does not say how wide the network's output is is fine: the network is run
  # on zero rows, so nothing has to be derived from `output_dim` (`tt_task()` fills one in, so this
  # task is built directly)
  no_dim = as_task_torch(d, target = "y", id = "n", default_encoder = tt_enc)
  expect_null(no_dim$output_dim)
  empty = create_empty_prediction_data(no_dim, learner)
  expect_numeric(empty$response, len = 0L)

  # without a model there is no network to ask, and `output_dim` is not a substitute: it describes
  # a single tensor, which is not what a network with more than one head returns
  untrained = tt_learner(t_loss("mse"))
  expect_error(create_empty_prediction_data(no_dim, untrained), "has no model")
  expect_error(create_empty_prediction_data(tt_task(d, target = "y"), untrained), "has no model")
})

test_that("the empty prediction never builds an empty batch", {
  d = tt_data(20L)
  d$y = rnorm(nrow(d))
  # plenty of real networks and batchgetters cannot be run on zero rows, so the structure is taken
  # from a batch of one row and cut back afterwards
  picky = nn_module("picky",
    initialize = function(task) self$l = nn_linear(length(task$feature_names), 1L),
    forward = function(x) {
      if (!nrow(x)) stop("this network refuses empty batches")
      self$l(x)
    })
  # no `output_dim`, so a failure here cannot hide behind the fallback
  task = as_task_torch(d, target = "y", id = "picky", default_encoder = tt_enc)
  learner = tt_learner(t_loss("mse"), module_generator = picky)
  learner$train(task)

  empty = create_empty_prediction_data(task, learner)
  expect_numeric(empty$response, len = 0L)
  expect_length(learner$predict(task, row_ids = integer(0))$row_ids, 0L)
})

test_that("an empty prediction is encoded from what the network really returns", {
  task = tt_task_2head()
  learner = tt_learner(tt_loss_2head(), module_generator = tt_module_2head,
    predict_types = c("response", "se"), predict_type = "se")
  learner$train(task)

  # the encoder is handed the `list()` of tensors it is promised, the same as for a real prediction
  empty = create_empty_prediction_data(task, learner)
  expect_names(names(empty), permutation.of = c("row_ids", "truth", "response", "se"))
  expect_numeric(empty$response, len = 0L)
  expect_numeric(empty$se, len = 0L)

  # ... so the two combine, which is what an empty resampling fold does
  expect_length(c(empty, learner$predict(task)$data)$response, task$nrow)
  rr = resample(task, learner, rsmp("custom")$instantiate(task,
    list(task$row_ids, task$row_ids), list(task$row_ids, integer(0))))
  expect_length(rr$prediction()$row_ids, task$nrow)
})

test_that("an encoder that does not return a named list errors here", {
  d = tt_data(10L)
  d$y = rnorm(nrow(d))
  encoders = list(
    tensor = function(task, network_output, predict_type) network_output,
    null = function(task, network_output, predict_type) NULL,
    unnamed = function(task, network_output, predict_type) list(1, 2)
  )
  for (nm in names(encoders)) {
    task = tt_task(d, target = "y", id = "bad", default_encoder = encoders[[nm]])
    learner = tt_learner(t_loss("mse"))
    learner$train(task)
    # the message names the task and the encoder, which the downstream `cannot coerce type
    # 'externalptr'` and `attempt to set an attribute on NULL` did not
    expect_error(learner$predict(task), "prediction encoding of task 'bad'", info = nm)
  }
})

test_that("prediction data can be combined and filtered", {
  d = tt_data(10L)
  d$y = rnorm(nrow(d))
  task = tt_task(d, target = "y")

  p1 = PredictionTorch$new(task, row_ids = 1:5, truth = task$truth(1:5), response = rnorm(5))
  p2 = PredictionTorch$new(task, row_ids = 6:10, truth = task$truth(6:10), response = rnorm(5))

  combined = c(p1$data, p2$data)
  expect_class(combined, "PredictionDataTorch")
  expect_equal(combined$row_ids, 1:10)
  expect_numeric(combined$response, len = 10L)

  filtered = filter_prediction_data(combined, 3:6)
  expect_equal(filtered$row_ids, 3:6)
  expect_numeric(filtered$response, len = 4L)

  tab = as.data.table(as_prediction(combined))
  expect_data_table(tab, nrows = 10L)
  expect_names(names(tab), identical.to = c("row_ids", "truth", "response"))
})

test_that("prediction data checks that everything describes the same observations", {
  d = tt_data(10L)
  d$y = rnorm(nrow(d))
  task = tt_task(d, target = "y")
  expect_error(
    PredictionTorch$new(task, row_ids = 1:5, truth = task$truth(1:5), response = rnorm(4)),
    "has 4 observations"
  )
})

test_that("matrix valued predictions survive a resample round trip", {
  d = tt_data(60L)
  d$y1 = rnorm(nrow(d))
  d$y2 = rnorm(nrow(d))
  task = tt_task(d, target = c("y1", "y2"))

  rr = resample(task, tt_learner(t_loss("mse")), rsmp("cv", folds = 3L))
  pred = rr$prediction()
  expect_matrix(pred$response, nrows = task$nrow, ncols = 2L)
  expect_equal(colnames(pred$response), c("y1", "y2"))
  expect_data_table(pred$truth, nrows = task$nrow, ncols = 2L)

  # a matrix response is one cell per observation, like an array of any other dimensionality: only
  # `prob` becomes one column per class, because only there does a column mean something fixed
  tab = as.data.table(pred)
  expect_class(tab$response, "pt_arrays")
  expect_equal(format(tab$response)[1L], "<array[2]>")
  # the cell is a one-dimensional array, so compare the values it holds
  expect_equal(as.numeric(tab$response[[1L]]), unname(pred$response[1L, ]))
  expect_false(any(startsWith(names(tab), "response.")))
})

test_that("only a prob matrix spreads into one column per class", {
  d = tt_data(10L)
  d$y = rnorm(nrow(d))
  task = tt_task(d, target = "y")

  # two dimensions are what a column per class means, so this is the one shape that spreads
  flat = PredictionTorch$new(task, response = rnorm(task$nrow),
    prob = matrix(runif(task$nrow * 2L), ncol = 2L, dimnames = list(NULL, c("a", "b"))))
  expect_names(names(as.data.table(flat)), must.include = c("prob.a", "prob.b"))

  # anything wider is one cell per observation, like every other wide element -- spreading it would
  # give a column per class *and* per pixel
  wide = PredictionTorch$new(task, response = rnorm(task$nrow),
    prob = array(runif(task$nrow * 2L * 3L), dim = c(task$nrow, 2L, 3L)))
  tab = as.data.table(wide)
  expect_class(tab$prob, "pt_arrays")
  expect_equal(format(tab$prob)[1L], "<array[2x3]>")
  expect_false(any(startsWith(names(tab), "prob.")))

  # a `prob` that is one value per observation stays one column
  vec = PredictionTorch$new(task, response = rnorm(task$nrow), prob = runif(task$nrow))
  expect_true("prob" %chin% names(as.data.table(vec)))
})

test_that("factor valued predictions survive a resample round trip", {
  d = tt_data(60L)
  d$y = factor(rep(c("a", "b", "c"), length.out = nrow(d)))
  task = tt_task(d, target = "y")

  rr = resample(task, tt_learner(t_loss("cross_entropy")), rsmp("cv", folds = 3L))
  pred = rr$prediction()
  expect_factor(pred$response, levels = c("a", "b", "c"), len = task$nrow)
  expect_factor(pred$truth, levels = c("a", "b", "c"), len = task$nrow)
})

test_that("array valued predictions survive a resample round trip", {
  d = tt_data(60L)
  for (nm in c("y1", "y2", "y3", "y4")) d[[nm]] = rnorm(nrow(d))
  # the prediction of an observation is a (2, 2) array rather than a vector, which is the shape an
  # autoencoder over images produces
  task = tt_task(d, target = c("y1", "y2", "y3", "y4"),
    default_encoder = function(task, network_output, predict_type) {
      x = as.array(network_output$cpu())
      list(response = array(x, dim = c(nrow(x), 2L, 2L)))
    })

  rr = resample(task, tt_learner(t_loss("mse")), rsmp("cv", folds = 3L))
  pred = rr$prediction()
  expect_array(pred$response, mode = "numeric", d = 3L)
  expect_equal(dim(pred$response), c(task$nrow, 2L, 2L))
  expect_data_table(pred$truth, nrows = task$nrow, ncols = 4L)

  # one row per observation, each holding its own array: flattening the cells into columns would
  # scale with the size of the array, and an autoencoder over images has 150528 of them
  tab = as.data.table(pred)
  # row_ids, one column per target of the truth, and one for the response
  expect_data_table(tab, nrows = task$nrow, ncols = 6L)
  expect_class(tab$response, "pt_arrays")
  expect_equal(dim(tab$response[[1L]]), c(2L, 2L))
  expect_equal(tab$response[[1L]], pred$response[1L, , ])

  # ... and it prints as the shape rather than as every value of every observation
  expect_match(format(tab$response)[1L], "<array[2x2]>", fixed = TRUE)
  expect_true(sum(nchar(capture.output(print(pred)))) < 1000L)
})

test_that("only a response with one value per observation reports missing predictions", {
  pdata = function(response) {
    structure(list(row_ids = 1:3, response = response),
      class = c("PredictionDataTorch", "PredictionData"))
  }

  # one value per observation: an NA is an observation that was not predicted
  expect_equal(is_missing_prediction_data(pdata(c(1, NA, 3))), 2L)
  expect_equal(is_missing_prediction_data(pdata(factor(c("a", NA, "c")))), 2L)
  expect_equal(is_missing_prediction_data(pdata(c(1, 2, 3))), integer(0))

  # anything wider does not: what a partially missing observation means is the encoder's business,
  # and a `lazy_tensor` could only answer by materialising the whole prediction
  a = array(1, dim = c(3L, 2L, 2L))
  a[2L, 2L, 1L] = NA
  expect_equal(is_missing_prediction_data(pdata(a)), integer(0))
  expect_equal(is_missing_prediction_data(pdata(matrix(c(1, NA, 3, 4, 5, 6), nrow = 3L))), integer(0))
  expect_equal(is_missing_prediction_data(pdata(data.table(a = c(1, NA, 3)))), integer(0))
  expect_equal(is_missing_prediction_data(pdata(as_lazy_tensor(torch_randn(3L, 2L)))), integer(0))

  # a prediction without a response has nothing that could be missing
  expect_equal(is_missing_prediction_data(pdata(NULL)), integer(0))
})

test_that("a lazy_tensor target survives a resample round trip", {
  d = tt_data(32L)
  d$y = as_lazy_tensor(withr::with_seed(2L, torch_randn(nrow(d), 3L)))
  task = tt_task(d, target = "y", id = "lt",
    output_dim = function(task) 3L,
    default_encoder = function(task, network_output, predict_type) {
      list(response = as.matrix(network_output$cpu()))
    })
  learner = tt_learner(t_loss("mse"),
    target_batchgetter = function(data) materialize(data[[1L]], rbind = TRUE)$to(dtype = torch_float()))

  rr = resample(task, learner, rsmp("cv", folds = 2L))
  pred = rr$prediction()
  # `unlist()` used to flatten the lazy_tensor into its internals, doubling its length
  expect_class(pred$truth, "lazy_tensor")
  expect_length(pred$truth, task$nrow)
  expect_data_table(as.data.table(pred), nrows = task$nrow)
  expect_equal(
    as.matrix(materialize(pred$truth, rbind = TRUE)),
    as.matrix(materialize(task$truth(pred$row_ids), rbind = TRUE))
  )
})

test_that("combining prediction data keeps the storage of its elements", {
  # the fallback of `pt_combine()` must not strip a class it does not know about
  lt = as_lazy_tensor(withr::with_seed(3L, torch_randn(4L, 2L)))
  combined = pt_combine(list(lt[1:2], lt[3:4]))
  expect_class(combined, "lazy_tensor")
  expect_length(combined, 4L)

  dates = pt_combine(list(as.Date("2020-01-01"), as.Date("2020-01-02")))
  expect_class(dates, "Date")
  expect_equal(dates, as.Date(c("2020-01-01", "2020-01-02")))

  expect_equal(pt_combine(list(list(1:2, 3:4), list(5:6))), list(1:2, 3:4, 5:6))
})

test_that("combining prediction data unions the levels of a factor", {
  # a resampling fold need not see every class, and its prediction then carries fewer levels than
  # the others; going by the levels of the first element would silently turn the rest into NA
  a = factor(c("a", "b", "a"), levels = c("a", "b"))
  b = factor(c("c", "b", "c"), levels = c("b", "c"))
  expect_equal(pt_combine(list(a, b)), factor(c("a", "b", "a", "c", "b", "c"), levels = c("a", "b", "c")))
  expect_equal(pt_combine(list(b, a)), factor(c("c", "b", "c", "a", "b", "a"), levels = c("b", "c", "a")))

  # the same, through the operation that a `resample()` over such folds performs
  pa = list(row_ids = 1:3, response = a)
  pb = list(row_ids = 4:6, response = b)
  class(pa) = class(pb) = c("PredictionDataTorch", "PredictionData")
  expect_equal(as.character(c(pa, pb)$response), c("a", "b", "a", "c", "b", "c"))

  # an ordered factor stays ordered, as long as the elements agree on the order
  oa = ordered(c("lo", "mid"), levels = c("lo", "mid", "hi"))
  ob = ordered("hi", levels = c("mid", "hi"))
  expect_equal(pt_combine(list(oa, ob)), ordered(c("lo", "mid", "hi"), levels = c("lo", "mid", "hi")))
  # ... and elements that disagree about the order are demoted to a plain factor, loudly, which is
  # `rbindlist()`'s answer to an ambiguity it cannot resolve
  expect_warning(
    demoted <- pt_combine(list(oa, ordered("hi", levels = c("hi", "mid")))),
    "ambiguity"
  )
  expect_false(is.ordered(demoted))
  expect_equal(as.character(demoted), c("lo", "mid", "hi"))
  expect_set_equal(levels(demoted), c("lo", "mid", "hi"))
})

test_that("combining prediction data with inconsistent elements is an error", {
  a = list(row_ids = 1:2, response = c(1, 2))
  b = list(row_ids = 3:4, response = c(3, 4))
  class(a) = class(b) = c("PredictionDataTorch", "PredictionData")
  expect_equal(length(c(a, b)$row_ids), 4L)

  # mlr3's own combine path does not check, so `c()` has to
  broken = b
  broken$response = c(3, 4, 5)
  expect_error(c(a, broken), "has 5 observations, but 4 row ids")
})

test_that("as.data.table refuses a prediction whose elements disagree", {
  # `cbind()` recycles a shorter table instead of complaining, which would turn a malformed
  # prediction into a table with duplicated observations rather than an error
  fitted = tt_fitted(10L)
  pred = fitted$prediction
  pred$data$response = pred$data$response[1:5]

  expect_error(as.data.table(pred), "has 10 row ids, but its elements have")
})

test_that("an array column prints as its shape rather than its contents", {
  cells = pt_arrays(array(seq_len(2L * 3L * 4L), c(2L, 3L, 4L)))

  expect_class(cells, "pt_arrays")
  expect_length(cells, 2L)
  expect_equal(dim(cells[[1L]]), c(3L, 4L))
  expect_equal(format(cells), rep("<array[3x4]>", 2L))

  # `data.table` pastes the contents of a list column unless its class has a `format_col()` method,
  # which for a batch of images is megabytes of numbers per printed prediction
  tab = data.table(x = cells)
  expect_match(capture.output(print(tab))[3L], "<array[3x4]>", fixed = TRUE)

  # an empty prediction has no cells to format
  expect_length(pt_arrays(array(numeric(0), c(0L, 3L, 4L))), 0L)
  expect_equal(format(pt_arrays(array(numeric(0), c(0L, 3L, 4L)))), character(0))
})

test_that("measure weights are subset and combined with their observations", {
  d = tt_data(40L)
  d$a = d$x1 > 0
  d$b = d$x2 > 0
  d$w = seq_len(nrow(d)) * 1.0
  task = tt_task(d, target = c("a", "b"), id = "w")
  task$set_col_roles("w", "weights_measure")

  learner = tt_learner(tt_loss_bce())
  learner$train(task)
  pred = learner$predict(task)
  expect_equal(pred$data$weights, d$w)

  # `weights` is not a predict type, but it describes the observations and has to travel with them
  keep = task$row_ids[c(2L, 4L, 6L)]
  filtered = filter_prediction_data(pred$data, row_ids = keep)
  expect_equal(filtered$weights, d$w[keep])

  combined = c(pred$data, pred$data, keep_duplicates = FALSE)
  expect_equal(combined$weights, d$w)

  # ... and an empty prediction carries the element, so that it can be combined with a real one
  empty = create_empty_prediction_data(task, learner)
  expect_equal(empty$weights, numeric(0))
  expect_length(c(empty, pred$data)$weights, task$nrow)
})

test_that("prediction data without a response has no missing predictions", {
  # a task can be scored by a measure that reads the truth from the task, so an encoder need not
  # produce a `response` at all -- there is then nothing that could be missing
  pdata = list(row_ids = 1:3, truth = c(1, 2, 3))
  class(pdata) = c("PredictionDataTorch", "PredictionData")

  expect_equal(is_missing_prediction_data(pdata), integer(0))
  expect_equal(check_prediction_data(pdata), pdata)
  expect_equal(c(pdata, pdata)$row_ids, c(1:3, 1:3))
})

test_that("combining degenerate collections of prediction data", {
  fitted = tt_fitted(10L)
  pdata = fitted$prediction$data
  empty = create_empty_prediction_data(fitted$task, fitted$learner)

  # a single element comes back as it is
  expect_equal(pt_combine(list(1:3)), 1:3)
  expect_equal(c(pdata)$row_ids, pdata$row_ids)

  # elements without observations carry no information about the storage, so they are dropped --
  # unless there is nothing else, in which case the result is still empty and still typed
  expect_equal(pt_combine(list(numeric(0), c(1, 2))), c(1, 2))
  expect_equal(pt_combine(list(numeric(0), numeric(0))), numeric(0))
  expect_equal(NROW(c(empty, empty)$response), 0L)
  expect_equal(class(c(empty, empty)$response), class(pdata$response))

  # `NULL` elements are dropped before anything looks at the storage
  expect_equal(pt_combine(list(NULL, c(1, 2), NULL)), c(1, 2))
})

test_that("filtering prediction data to nothing keeps every element", {
  fitted = tt_fitted(10L)
  pdata = fitted$prediction$data

  gone = filter_prediction_data(pdata, row_ids = integer(0))
  expect_set_equal(names(gone), names(pdata))
  expect_equal(gone$row_ids, integer(0))
  expect_equal(NROW(gone$response), 0L)

  # row ids that are not in the prediction are not an error, they simply match nothing
  expect_equal(filter_prediction_data(pdata, row_ids = 1000L)$row_ids, integer(0))

  # ... and the emptied prediction still combines with a real one
  expect_equal(NROW(c(gone, pdata)$response), NROW(pdata$response))
})

test_that("every storage an element can have is subset by observation", {
  # `pt_subset()` is what filtering and dropping duplicates go through, and the elements of a
  # `PredictionTorch` may be stored in any of these ways
  expect_equal(pt_subset(c(10, 20, 30), c(1L, 3L)), c(10, 30))
  expect_equal(pt_subset(matrix(1:6, nrow = 3L), c(1L, 3L)), matrix(c(1L, 3L, 4L, 6L), nrow = 2L))
  expect_equal(pt_subset(data.table(a = 1:3, b = 4:6), 2L), data.table(a = 2L, b = 5L))
  expect_equal(pt_subset(factor(c("a", "b", "c")), 2L), factor("b", levels = c("a", "b", "c")))

  a = array(1:24, c(3L, 2L, 4L))
  expect_equal(dim(pt_subset(a, c(1L, 2L))), c(2L, 2L, 4L))
  expect_equal(pt_subset(a, 2L)[1L, , ], a[2L, , ])

  # a matrix keeps its column names, which for a `prob` element are the class labels
  m = matrix(1:6, nrow = 3L, dimnames = list(NULL, c("a", "b")))
  expect_equal(colnames(pt_subset(m, 1L)), c("a", "b"))
})

test_that("the lazy_tensor predict type hands back the network output", {
  task = tt_task_raw()
  learner = tt_learner(t_loss("mse"))

  # unlike `prob` and `se` it is not opt-in: every learner for this task type can hand the
  # network's output back, so nothing has to be declared at construction
  expect_set_equal(learner$predict_types, c("response", "lazy_tensor"))

  learner$predict_type = "lazy_tensor"
  learner$train(task)
  pred = learner$predict(task)

  expect_equal(pred$predict_types, "lazy_tensor")
  expect_class(pred$lazy_tensor, "lazy_tensor")
  expect_length(pred$lazy_tensor, task$nrow)
  expect_null(pred$response)
  expect_equal(materialize(pred$lazy_tensor, rbind = TRUE)$shape, c(task$nrow, 1L))

  # it is what the network produced, not something the task derived
  batch = learner$dataset(task)$.getbatch(seq_len(task$nrow))
  learner$network$eval()
  expected = with_no_grad(learner$network(batch$x$x))
  # one batch of 20 rows and the two batches `$predict()` used do not take the same path through
  # float32 arithmetic, so this is equal only up to rounding
  expect_equal(
    as.numeric(as.matrix(materialize(pred$lazy_tensor, rbind = TRUE)$cpu())),
    as.numeric(as.matrix(expected$cpu())), tolerance = 1e-6
  )

  # ... and it travels through the machinery like any other element
  expect_true("lazy_tensor" %chin% names(as.data.table(pred)))
  expect_match(capture.output(print(pred))[4L], "<tnsr[1]>", fixed = TRUE)
  pred$filter(task$row_ids[1:5])
  expect_length(pred$lazy_tensor, 5L)
})

test_that("lazy_tensor predictions of different folds are combined", {
  task = tt_task_raw()
  learner = tt_learner(t_loss("mse"))
  learner$predict_type = "lazy_tensor"

  # every fold wraps its own network output, so the folds share no data descriptor and
  # `c.lazy_tensor()` alone would refuse them
  rr = resample(task, learner, rsmp("cv", folds = 2L))
  pred = rr$prediction()
  expect_length(pred$lazy_tensor, task$nrow)
  expect_equal(materialize(pred$lazy_tensor, rbind = TRUE)$shape, c(task$nrow, 1L))

  # each fold's rows keep the values that fold predicted
  first = rr$predictions()[[1L]]
  combined = as.numeric(as.matrix(materialize(pred$lazy_tensor, rbind = TRUE)$cpu()))
  expect_equal(
    combined[match(first$row_ids, pred$row_ids)],
    as.numeric(as.matrix(materialize(first$lazy_tensor, rbind = TRUE)$cpu()))
  )

  # only lazy tensors that already hold their data are combined this way: materialising ones that
  # read on demand would pull whole datasets into memory, so they are refused
  on_demand = as_lazy_tensor(
    torch::dataset("on_demand",
      initialize = function() NULL,
      .getbatch = function(ids) list(x = torch_randn(length(ids), 1L)),
      .length = function() 5L
    )(),
    dataset_shapes = list(x = c(NA, 1L))
  )
  expect_false(inherits(dd(on_demand)$dataset, "in_memory_tensor_dataset"))
  expect_error(pt_combine(list(on_demand, pred$lazy_tensor)), "reads its data on demand")

  # an empty prediction is an empty lazy tensor, because one cannot be encoded from zero rows.
  # `resample()` clones, so this learner still has to be trained to have a network to ask
  learner$train(task)
  empty = create_empty_prediction_data(task, learner)
  expect_class(empty$lazy_tensor, "lazy_tensor")
  expect_length(empty$lazy_tensor, 0L)
  expect_length(c(empty, pred$data)$lazy_tensor, task$nrow)
})

test_that("a lazy_tensor prediction of a two-head network is one column per head", {
  task = tt_task_2head()
  learner = tt_learner(tt_loss_2head(), module_generator = tt_module_2head)
  learner$predict_type = "lazy_tensor"
  learner$train(task)
  pred = learner$predict(task)

  # one `lazy_tensor` per head, so that the prediction is still one row per observation
  expect_data_table(pred$lazy_tensor, nrows = task$nrow, ncols = 2L)
  expect_names(names(pred$lazy_tensor), identical.to = c("m", "s"))
  expect_class(pred$lazy_tensor$m, "lazy_tensor")

  # it is what the network produced, head by head
  batch = learner$dataset(task)$.getbatch(seq_len(task$nrow))
  learner$network$eval()
  tensors = with_no_grad(learner$network(batch$x$x))
  for (head in c("m", "s")) {
    expect_equal(
      as.numeric(as.matrix(materialize(pred$lazy_tensor[[head]], rbind = TRUE)$cpu())),
      as.numeric(as.matrix(tensors[[head]]$cpu())), tolerance = 1e-6
    )
  }

  # ... and `as.data.table()` spreads it, the way a `data.table` truth spreads
  expect_names(names(as.data.table(pred)), must.include = c("lazy_tensor.m", "lazy_tensor.s"))

  # combining keeps every row with the values of its own fold, head-wise
  parts = c(learner$predict(task, row_ids = 11:20)$data, learner$predict(task, row_ids = 1:10)$data)
  expect_data_table(parts$lazy_tensor, nrows = task$nrow, ncols = 2L)
  ord = match(task$row_ids, parts$row_ids)
  expect_equal(
    as.numeric(as.matrix(materialize(parts$lazy_tensor$m[ord], rbind = TRUE)$cpu())),
    as.numeric(as.matrix(materialize(pred$lazy_tensor$m, rbind = TRUE)$cpu()))
  )

  # an empty prediction agrees on the heads, so it combines with a real one
  empty = create_empty_prediction_data(task, learner)
  expect_data_table(empty$lazy_tensor, nrows = 0L, ncols = 2L)
  expect_length(c(empty, pred$data)$row_ids, task$nrow)

  pred$filter(task$row_ids[1:5])
  expect_data_table(pred$lazy_tensor, nrows = 5L, ncols = 2L)
})

test_that("the lazy_tensor predict type can be scored during training", {
  task = tt_task_raw(30L)
  measure = msr_torch("ltnorm", function(lazy_tensor) {
    as.numeric(materialize(lazy_tensor, rbind = TRUE)$pow(2)$mean()$cpu())
  }, predict_type = "lazy_tensor", range = c(0, Inf), minimize = TRUE)

  learner = tt_learner(t_loss("mse"), measures_train = measure, measures_valid = measure,
    patience = 2L, epochs = 3L)
  learner$predict_type = "lazy_tensor"
  learner$validate = 0.3
  learner$train(task)

  # a real number, not the `NaN` of a measure whose predict type the prediction never had -- which
  # is also what early stopping and internal tuning read, and a `NaN` there tunes `epochs` to 1
  expect_number(learner$internal_valid_scores$ltnorm, lower = 0)
  expect_number(learner$internal_tuned_values$epochs, lower = 1)
})
