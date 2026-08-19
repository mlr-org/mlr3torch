# a small MLP whose output layer is sized through `output_dim_for()`
tt_module = nn_module("tt_module",
  initialize = function(task) {
    self$net = nn_sequential(
      nn_linear(length(task$feature_names), 10L), nn_relu(),
      nn_linear(10L, output_dim_for(task))
    )
  },
  forward = function(x) self$net(x)
)

tt_learner = function(loss, ...) {
  args = insert_named(list(epochs = 3L, batch_size = 16L), list(...))
  invoke(lrn, "torch.module",
    module_generator = tt_module,
    ingress_tokens = list(x = ingress_num()),
    loss = loss,
    .args = args
  )
}

tt_data = function(n = 40L) {
  withr::with_seed(1L, data.table(
    x1 = rnorm(n), x2 = rnorm(n), x3 = rnorm(n)
  ))
}

test_that("the task type is registered", {
  expect_true("torch" %chin% mlr_reflections$task_types$type)
  expect_equal(
    mlr_reflections$task_types[list("torch"), "task", on = "type"][[1L]],
    "TaskTorch"
  )
  expect_set_equal(names(mlr_reflections$learner_predict_types$torch), c("response", "prob"))
  expect_equal(mlr_reflections$default_measures$torch, "torch.default")
})

test_that("as_task_torch constructs a TaskTorch", {
  d = tt_data()
  d$y = d$x1 + rnorm(nrow(d))
  task = as_task_torch(d, target = "y", id = "t")

  expect_class(task, "TaskTorch")
  expect_equal(task$task_type, "torch")
  expect_equal(task$id, "t")
  expect_set_equal(task$feature_names, c("x1", "x2", "x3"))
  expect_equal(task$target_names, "y")
})

test_that("a single factor target is inferred as multiclass", {
  d = tt_data()
  d$y = factor(rep(c("a", "b", "c"), length.out = nrow(d)))
  task = as_task_torch(d, target = "y")

  expect_equal(output_dim_for(task), 3L)
  expect_factor(task$truth(), levels = c("a", "b", "c"))

  y = get_target_batchgetter(task)(data.table(y = task$truth(1:4)))
  expect_equal(y$dtype, torch_long())
  expect_equal(as.integer(y), as.integer(task$truth(1:4)))

  # a two-level factor is *not* treated as binary classification
  d$y2 = factor(rep(c("a", "b"), length.out = nrow(d)))
  expect_equal(output_dim_for(as_task_torch(d[, c("x1", "y2")], target = "y2")), 2L)
})

test_that("numeric targets are inferred", {
  d = tt_data()
  d$y1 = rnorm(nrow(d))
  d$y2 = rnorm(nrow(d))

  single = as_task_torch(d[, c("x1", "y1")], target = "y1")
  expect_equal(output_dim_for(single), 1L)
  expect_numeric(single$truth())
  expect_equal(get_target_batchgetter(single)(data.table(y1 = 1:3))$shape, c(3L, 1L))

  multi = as_task_torch(d, target = c("y1", "y2"))
  expect_equal(output_dim_for(multi), 2L)
  expect_data_table(multi$truth(), ncols = 2L)
  expect_equal(get_target_batchgetter(multi)(data.table(a = 1:3, b = 4:6))$shape, c(3L, 2L))
})

test_that("logical targets are inferred", {
  d = tt_data()
  d$a = d$x1 > 0
  d$b = d$x2 > 0
  task = as_task_torch(d, target = c("a", "b"))

  expect_equal(output_dim_for(task), 2L)
  y = get_target_batchgetter(task)(data.table(a = c(TRUE, FALSE), b = c(FALSE, FALSE)))
  expect_equal(y$dtype, torch_float())
  expect_equal(as.matrix(y), matrix(c(1, 0, 0, 0), nrow = 2L))
})

test_that("an unsupported combination of target types errors informatively", {
  d = tt_data()
  d$y1 = rnorm(nrow(d))
  d$y2 = letters[seq_len(nrow(d))]
  task = as_task_torch(d, target = c("y1", "y2"))

  expect_error(output_dim_for(task), "pass `output_dim` explicitly")
  expect_error(get_target_batchgetter(task), "pass `target_batchgetter` explicitly")
  expect_error(encode_prediction(task, torch_randn(2, 2), "response"), "pass `prediction_encoder` explicitly")
})

test_that("the inferred fields can be overwritten", {
  d = tt_data()
  d$y1 = rnorm(nrow(d))
  d$y2 = letters[seq_len(nrow(d))]

  batchgetter = function(data) torch_tensor(matrix(0, nrow(data), 7L))
  encoder = function(task, predict_tensor, predict_type) list(response = rep(1, nrow(predict_tensor)))
  task = as_task_torch(d, target = c("y1", "y2"), output_dim = 7L,
    target_batchgetter = batchgetter, prediction_encoder = encoder)

  expect_equal(output_dim_for(task), 7L)
  expect_identical(get_target_batchgetter(task), batchgetter)
  expect_equal(encode_prediction(task, torch_randn(3, 7), "response")$response, rep(1, 3))
})

test_that("a learner can overwrite the target encoding of the task", {
  d = tt_data(60L)
  d$y = factor(rep(c("a", "b", "c"), length.out = nrow(d)))
  task = as_task_torch(d, target = "y")

  # the task encodes the target as class indices, but we train on one-hot labels with MSE instead
  onehot = function(data) {
    torch_tensor(1 * stats::model.matrix(~ 0 + data[[1L]]), dtype = torch_float())
  }
  learner = tt_learner(t_loss("mse"), target_batchgetter = onehot)
  expect_class(learner$dataset(task)$.getbatch(1:4)$y, "torch_tensor")
  expect_equal(learner$dataset(task)$.getbatch(1:4)$y$shape, c(4L, 3L))

  # the task's default is still the class indices
  expect_equal(get_target_batchgetter(task)(data.table(y = task$truth(1:4)))$dtype, torch_long())

  learner$train(task)
  expect_factor(learner$predict(task)$response, levels = c("a", "b", "c"), len = task$nrow)

  # the override is part of the learner's phash
  expect_false(learner$phash == tt_learner(t_loss("mse"))$phash)
})

test_that("the hash covers the fields that define the learning problem", {
  d = tt_data()
  d$y = rnorm(nrow(d))
  task = as_task_torch(d, target = "y", id = "t")

  expect_equal(task$hash, as_task_torch(d, target = "y", id = "t")$hash)
  expect_false(task$hash == as_task_torch(d, target = "y", id = "t", output_dim = 5L)$hash)
  expect_false(task$hash == as_task_torch(d, target = "y", id = "t",
    prediction_encoder = function(task, predict_tensor, predict_type) list(response = 1))$hash)
  expect_equal(task$hash, task$clone(deep = TRUE)$hash)
})

test_that("train, predict and score work for a multi-label task", {
  d = tt_data(60L)
  d$a = d$x1 > 0
  d$b = d$x2 > 0
  task = as_task_torch(d, target = c("a", "b"), id = "labels")

  learner = tt_learner(TorchLoss$new(torch::nn_bce_with_logits_loss, id = "bce"))
  learner$predict_type = "prob"
  learner$train(task)
  pred = learner$predict(task)

  expect_class(pred, "PredictionTorch")
  expect_matrix(pred$response, mode = "logical", nrows = task$nrow, ncols = 2L)
  expect_equal(colnames(pred$response), c("a", "b"))
  expect_matrix(pred$prob, mode = "numeric", nrows = task$nrow, ncols = 2L)
  expect_data_table(pred$truth, nrows = task$nrow, ncols = 2L)

  measure = msr_torch("hamming", function(truth, response) mean(as.matrix(truth) != response),
    range = c(0, 1))
  score = pred$score(measure)
  expect_number(score, lower = 0, upper = 1)
  expect_equal(names(score), "hamming")
})

test_that("train and predict work for a single factor target", {
  d = tt_data(60L)
  d$y = factor(rep(c("a", "b", "c"), length.out = nrow(d)))
  task = as_task_torch(d, target = "y")

  learner = tt_learner(t_loss("cross_entropy"))
  learner$predict_type = "prob"
  learner$train(task)
  pred = learner$predict(task)

  expect_factor(pred$response, levels = c("a", "b", "c"), len = task$nrow)
  expect_matrix(pred$prob, nrows = task$nrow, ncols = 3L)
  expect_equal(colnames(pred$prob), c("a", "b", "c"))
  expect_equal(unname(rowSums(pred$prob)), rep(1, task$nrow), tolerance = 1e-5)
})

test_that("asking for probabilities on a numeric target errors", {
  d = tt_data(40L)
  d$y = rnorm(nrow(d))
  task = as_task_torch(d, target = "y")
  expect_error(encode_prediction(task, torch_randn(4, 1), "prob"), "only predict_type 'response'")
})

test_that("train and predict work for a single numeric target", {
  d = tt_data(60L)
  d$y = d$x1 + rnorm(nrow(d))
  task = as_task_torch(d, target = "y")

  learner = tt_learner(t_loss("mse"))
  learner$train(task)
  pred = learner$predict(task)

  expect_numeric(pred$response, len = task$nrow)
  expect_numeric(pred$truth, len = task$nrow)
})

test_that("resampling works", {
  d = tt_data(60L)
  d$y = d$x1 + rnorm(nrow(d))
  task = as_task_torch(d, target = "y")
  measure = msr_torch("mse", function(truth, response) mean((truth - response)^2), range = c(0, Inf))

  rr = resample(task, tt_learner(t_loss("mse")), rsmp("cv", folds = 3L))
  expect_number(rr$aggregate(measure), lower = 0)
  # the predictions of the folds are combined and re-split, which exercises
  # c.PredictionDataTorch() and filter_prediction_data.PredictionDataTorch()
  expect_numeric(rr$prediction()$response, len = task$nrow)
  expect_set_equal(rr$prediction()$row_ids, task$row_ids)
})

test_that("validation and early stopping work", {
  d = tt_data(60L)
  d$y = d$x1 + rnorm(nrow(d))
  task = as_task_torch(d, target = "y")
  measure = msr_torch("mse", function(truth, response) mean((truth - response)^2), range = c(0, Inf))

  learner = tt_learner(t_loss("mse"), epochs = 20L, patience = 2L, measures_valid = measure)
  learner$validate = 0.3
  learner$train(task)

  expect_names(names(learner$internal_valid_scores), identical.to = "mse")
  expect_number(learner$internal_tuned_values$epochs, lower = 0, upper = 20L)
})

test_that("a target that is a function of the input needs no special support", {
  # an autoencoder is a target-less task whose batchgetter reads the input tensor
  d = tt_data(60L)
  task = as_task_torch(d, id = "ae", output_dim = 3L,
    target_batchgetter = function(data, x) x[[1L]],
    prediction_encoder = function(task, predict_tensor, predict_type) {
      response = as.matrix(predict_tensor$cpu())
      colnames(response) = task$feature_names
      list(response = response)
    })

  expect_equal(task$target_names, character(0))
  expect_equal(output_dim_for(task), 3L)
  expect_null(task$truth())

  # the batchgetter receives the feature tensors, so y is the input
  batch = task_dataset(task, list(x = ingress_num()),
    target_batchgetter = get_target_batchgetter(task))$.getbatch(1:4)
  expect_equal(as.matrix(batch$y), as.matrix(batch$x$x))

  learner = tt_learner(t_loss("mse"), epochs = 5L)
  learner$train(task)
  pred = learner$predict(task)

  expect_matrix(pred$response, nrows = task$nrow, ncols = 3L)
  expect_equal(colnames(pred$response), c("x1", "x2", "x3"))
  expect_false("truth" %chin% names(pred$data))

  # there is no truth, so the measure reads the ground truth from the task
  measure = msr_torch("recon", function(task, prediction) {
    truth = as.matrix(task$data(rows = prediction$row_ids, cols = task$feature_names))
    mean((truth - prediction$response)^2)
  }, range = c(0, Inf))
  expect_number(pred$score(measure, task = task), lower = 0)
  expect_number(resample(task, learner, rsmp("cv", folds = 3L))$aggregate(measure), lower = 0)
})

test_that("a task with no target at all is unsupervised", {
  d = tt_data(60L)
  task = as_task_torch(d, id = "unsup")

  expect_equal(task$target_names, character(0))
  expect_null(task$truth())
  expect_null(get_target_batchgetter(task))
  # nothing can be inferred for such a task
  expect_error(output_dim_for(task), "pass `output_dim` explicitly")
  expect_error(encode_prediction(task, torch_randn(2, 2), "response"), "pass `prediction_encoder` explicitly")

  task = as_task_torch(d, id = "unsup", output_dim = 2L,
    prediction_encoder = function(task, predict_tensor, predict_type) {
      list(response = as.matrix(predict_tensor$cpu()))
    })
  expect_equal(output_dim_for(task), 2L)
  # the batches carry no target
  expect_names(names(task_dataset(task, list(x = ingress_num()))$.getbatch(1:4)),
    permutation.of = c("x", ".index"))

  # the loss must ignore its second argument
  loss = TorchLoss$new(nn_module("spread",
    initialize = function() NULL,
    forward = function(input, target) input$pow(2)$mean()), id = "spread")
  learner = tt_learner(loss)
  learner$train(task)
  pred = learner$predict(task)

  expect_matrix(pred$response, nrows = task$nrow, ncols = 2L)
  expect_false("truth" %chin% names(pred$data))
  expect_null(pred$truth)

  # a measure cannot read a truth, so it reads the prediction or the task
  measure = msr_torch("spread", function(prediction) mean(prediction$response^2), range = c(0, Inf))
  expect_number(pred$score(measure), lower = 0)
  expect_number(resample(task, learner, rsmp("cv", folds = 3L))$aggregate(measure), lower = 0)
})

test_that("an unsupervised task works with validation and the tensor dataset", {
  d = tt_data(60L)
  task = as_task_torch(d, id = "unsup", output_dim = 2L,
    prediction_encoder = function(task, predict_tensor, predict_type) {
      list(response = as.matrix(predict_tensor$cpu()))
    })
  loss = TorchLoss$new(nn_module("spread",
    initialize = function() NULL,
    forward = function(input, target) input$pow(2)$mean()), id = "spread")
  measure = msr_torch("spread", function(prediction) mean(prediction$response^2), range = c(0, Inf))

  learner = tt_learner(loss, epochs = 10L, patience = 2L, measures_valid = measure)
  learner$validate = 0.3
  learner$train(task)
  expect_names(names(learner$internal_valid_scores), identical.to = "spread")

  learner = tt_learner(loss, tensor_dataset = TRUE)
  learner$train(task)
  expect_matrix(learner$predict(task)$response, nrows = task$nrow, ncols = 2L)
})

test_that("a measure can read the task", {
  d = tt_data(40L)
  d$y = rnorm(nrow(d))
  task = as_task_torch(d, target = "y")
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
  task = as_task_torch(d, target = "y")
  measure = msr_torch("seen", function(learner, train_set) {
    stopifnot(inherits(learner, "LearnerTorch"))
    length(train_set)
  })
  rr = resample(task, tt_learner(t_loss("mse")), rsmp("holdout", ratio = 0.5))
  expect_equal(unname(rr$aggregate(measure)), 20)
})

test_that("the default measure of a task is used by aggregate()", {
  d = tt_data(40L)
  d$y = rnorm(nrow(d))
  measure = msr_torch("mse", function(truth, response) mean((truth - response)^2), range = c(0, Inf))

  without = as_task_torch(d, target = "y", id = "t")
  rr = resample(without, tt_learner(t_loss("mse")), rsmp("holdout"))
  expect_error(rr$aggregate(), "has no default measure")

  with = as_task_torch(d, target = "y", id = "t", measure = measure)
  rr = resample(with, tt_learner(t_loss("mse")), rsmp("holdout"))
  expect_number(rr$aggregate(), lower = 0)
})

test_that("the graph language works", {
  d = tt_data(60L)
  d$y = d$x1 + rnorm(nrow(d))
  task = as_task_torch(d, target = "y")

  graph = po("torch_ingress_num") %>>%
    nn("linear_1", out_features = 10L) %>>%
    nn("relu_1") %>>%
    nn("head") %>>%
    po("torch_loss", t_loss("mse")) %>>%
    po("torch_optimizer", "adam") %>>%
    po("torch_model", batch_size = 16L, epochs = 3L)

  learner = as_learner(graph)
  learner$train(task)
  pred = learner$predict(task)
  expect_class(pred, "PredictionTorch")
  expect_numeric(pred$response, len = task$nrow)
})

test_that("a learner for a torch task requires an explicit loss", {
  expect_error(
    lrn("torch.module", module_generator = tt_module, ingress_tokens = list(x = ingress_num())),
    "no default loss"
  )
  # any loss is accepted, because mlr3torch cannot know what the task represents
  expect_class(tt_learner(t_loss("mse"))$loss, "TorchLoss")
  expect_class(tt_learner(t_loss("cross_entropy"))$loss, "TorchLoss")
})

test_that("a network output that does not match the target levels errors", {
  d = tt_data(60L)
  d$y = factor(rep(c("a", "b", "c"), length.out = nrow(d)))
  task = as_task_torch(d, target = "y")

  expect_error(encode_prediction(task, torch_randn(6, 5), "response"), "incompatible with the 3 levels")
  expect_error(encode_prediction(task, torch_randn(6, 2), "response"), "incompatible with the 3 levels")
  expect_error(encode_prediction(task, torch_randn(6), "response"), "incompatible with the 3 levels")
  expect_factor(encode_prediction(task, torch_randn(6, 3), "response")$response,
    levels = c("a", "b", "c"))

  # the case that motivated the check: predicting on a task whose levels were dropped would
  # otherwise relabel every observation silently
  learner = tt_learner(t_loss("cross_entropy"))
  learner$train(task)
  dropped = task$clone(deep = TRUE)$filter(which(d$y != "a"))$droplevels()
  expect_error(learner$predict(dropped), "Was the network trained on a task with different levels")
  expect_factor(learner$predict(task)$response, levels = c("a", "b", "c"))
})

test_that("combining prediction data with different predict types errors", {
  d = tt_data(6L)
  d$a = d$x1 > 0
  task = as_task_torch(d, target = "a", id = "t")

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
  task = function(measure) as_task_torch(d, target = "y", id = "same", measure = measure)
  expect_false(task(msr_torch("m", function(truth, response) 1))$hash ==
    task(msr_torch("m", function(truth, response) 2))$hash)

  # ... which is what stops benchmark() from scoring both rows with one task's measure
  bmr = benchmark(benchmark_grid(
    list(task(msr_torch("m", function(truth, response) 1)),
      task(msr_torch("m", function(truth, response) 999))),
    tt_learner(t_loss("mse")), rsmp("holdout")))
  expect_equal(bmr$aggregate(msr("torch.default"))$torch.default, c(1, 999))
})

test_that("predicting on zero rows gives an empty prediction", {
  d = tt_data(40L)
  d$a = d$x1 > 0
  d$b = d$x2 > 0
  task = as_task_torch(d, target = c("a", "b"), id = "t")
  learner = tt_learner(TorchLoss$new(torch::nn_bce_with_logits_loss, id = "bce"))
  learner$predict_type = "prob"
  learner$train(task)

  pred = learner$predict(task$clone(deep = TRUE)$filter(integer(0)))
  expect_class(pred, "PredictionTorch")
  expect_equal(length(pred$row_ids), 0L)

  # an empty prediction must have the same storage as a non-empty one, so the two can be combined
  empty = create_empty_prediction_data(task, learner)
  expect_names(names(empty), permutation.of = c("row_ids", "truth", "response", "prob"))
  expect_matrix(empty$response, nrows = 0L, ncols = 2L)

  combined = c(empty, learner$predict(task)$data)
  expect_matrix(combined$response, nrows = task$nrow, ncols = 2L)
  expect_matrix(combined$prob, nrows = task$nrow, ncols = 2L)
  expect_equal(combined$row_ids, task$row_ids)
})

test_that("prediction data can be combined and filtered", {
  d = tt_data(10L)
  d$y = rnorm(nrow(d))
  task = as_task_torch(d, target = "y")

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
  task = as_task_torch(d, target = "y")
  expect_error(
    PredictionTorch$new(task, row_ids = 1:5, truth = task$truth(1:5), response = rnorm(4)),
    "has 4 observations"
  )
})

test_that("matrix valued predictions survive a resample round trip", {
  d = tt_data(60L)
  d$y1 = rnorm(nrow(d))
  d$y2 = rnorm(nrow(d))
  task = as_task_torch(d, target = c("y1", "y2"))

  rr = resample(task, tt_learner(t_loss("mse")), rsmp("cv", folds = 3L))
  pred = rr$prediction()
  expect_matrix(pred$response, nrows = task$nrow, ncols = 2L)
  expect_equal(colnames(pred$response), c("y1", "y2"))
  expect_data_table(pred$truth, nrows = task$nrow, ncols = 2L)
})

test_that("factor valued predictions survive a resample round trip", {
  d = tt_data(60L)
  d$y = factor(rep(c("a", "b", "c"), length.out = nrow(d)))
  task = as_task_torch(d, target = "y")

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
  task = as_task_torch(d, target = c("y1", "y2", "y3", "y4"),
    prediction_encoder = function(task, predict_tensor, predict_type) {
      x = as.array(predict_tensor$cpu())
      list(response = array(x, dim = c(nrow(x), 2L, 2L)))
    })

  rr = resample(task, tt_learner(t_loss("mse")), rsmp("cv", folds = 3L))
  pred = rr$prediction()
  expect_array(pred$response, mode = "numeric", d = 3L)
  expect_equal(dim(pred$response), c(task$nrow, 2L, 2L))
  expect_data_table(pred$truth, nrows = task$nrow, ncols = 4L)

  # one row per observation, with the four cells of its array flattened into columns
  tab = as.data.table(pred)
  expect_data_table(tab, nrows = task$nrow)
  expect_equal(sum(startsWith(names(tab), "response")), 4L)
})

test_that("a learner for a different task type is rejected", {
  d = tt_data()
  d$y = d$x1 + rnorm(nrow(d))
  task = as_task_torch(d, target = "y")
  # `mlr3` accepts any learner inheriting the class registered for the task type, and `LearnerTorch`
  # is registered for "torch", so this mismatch has to be caught by `LearnerTorch` itself
  expect_error(
    lrn("regr.mlp", epochs = 1L, batch_size = 16L)$train(task),
    "is for task type 'regr'"
  )
})

test_that("a lazy_tensor target survives a resample round trip", {
  d = tt_data(32L)
  d$y = as_lazy_tensor(withr::with_seed(2L, torch_randn(nrow(d), 3L)))
  task = as_task_torch(d, target = "y", id = "lt",
    target_batchgetter = function(data) materialize(data[[1L]], rbind = TRUE)$to(dtype = torch_float()),
    output_dim = 3L,
    prediction_encoder = function(task, predict_tensor, predict_type) {
      list(response = as.matrix(predict_tensor$cpu()))
    })

  rr = resample(task, tt_learner(t_loss("mse")), rsmp("cv", folds = 2L))
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
