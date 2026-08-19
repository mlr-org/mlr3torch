test_that("the task type is registered", {
  expect_true("torch" %chin% mlr_reflections$task_types$type)
  expect_equal(
    mlr_reflections$task_types[list("torch"), "task", on = "type"][[1L]],
    "TaskTorch"
  )
  expect_set_equal(names(mlr_reflections$learner_predict_types$torch), c("response", "prob", "se"))
  expect_equal(mlr_reflections$default_measures$torch, "torch.default")
})

test_that("as_task_torch constructs a TaskTorch", {
  d = tt_data()
  d$y = d$x1 + rnorm(nrow(d))
  task = tt_task(d, target = "y", id = "t")

  expect_class(task, "TaskTorch")
  expect_equal(task$task_type, "torch")
  expect_equal(task$id, "t")
  expect_set_equal(task$feature_names, c("x1", "x2", "x3"))
  expect_equal(task$target_names, "y")
})

test_that("nothing about the learning problem is inferred", {
  d = tt_data()
  d$y = rnorm(nrow(d))
  # a plain task specifies none of the three, and each caller says where to set it
  bare = as_task_torch(d, target = "y", id = "bare")

  expect_error(output_dim_for(bare), "has no `output_dim`")
  expect_error(get_target_batchgetter(bare), "it is the learner that decides")
  expect_error(encode_prediction(bare, torch_randn(2, 1), "response"), "has no `prediction_encoder`")
})

test_that("output_dim is evaluated rather than stored", {
  d = tt_data()
  d$y1 = rnorm(nrow(d))
  d$y2 = rnorm(nrow(d))
  task = as_task_torch(d, target = c("y1", "y2"), id = "t",
    output_dim = function(task) length(task$target_names))

  expect_equal(output_dim_for(task), 2L)
  # a stored number would go stale here, which is why it has to be a function
  task$col_roles$target = "y1"
  expect_equal(output_dim_for(task), 1L)

  expect_error(as_task_torch(d, target = "y1", output_dim = 1L), "Must be a function")
})

test_that("encode_prediction dispatches to the task's encoder", {
  d = tt_data()
  d$y = rnorm(nrow(d))
  encoder = function(task, predict_tensor, predict_type) list(response = rep(1, nrow(predict_tensor)))
  task = as_task_torch(d, target = "y", id = "t", prediction_encoder = encoder)

  expect_equal(encode_prediction(task, torch_randn(3, 7), "response")$response, rep(1, 3))
  # the raw network output is handed over, so a multi-head network is the encoder's business
  multi = as_task_torch(d, target = "y", id = "t",
    prediction_encoder = function(task, predict_tensor, predict_type) {
      list(response = as.numeric(predict_tensor[[2L]]$cpu()))
    })
  expect_equal(encode_prediction(multi, list(torch_randn(3, 1), torch_ones(3, 1)), "response")$response, rep(1, 3))
})

test_that("the learner decides how the target becomes a tensor", {
  d = tt_data(60L)
  d$y = factor(rep(c("a", "b", "c"), length.out = nrow(d)))
  task = tt_task(d, target = "y")

  # train on one-hot labels with MSE rather than on class indices
  onehot = function(data) {
    torch_tensor(1 * stats::model.matrix(~ 0 + data[[1L]]), dtype = torch_float())
  }
  learner = tt_learner(t_loss("mse"), target_batchgetter = onehot)
  expect_class(learner$dataset(task)$.getbatch(1:4)$y, "torch_tensor")
  expect_equal(learner$dataset(task)$.getbatch(1:4)$y$shape, c(4L, 3L))

  learner$train(task)
  expect_factor(learner$predict(task)$response, levels = c("a", "b", "c"), len = task$nrow)

  # the batchgetter is part of the learner's phash
  expect_false(learner$phash == tt_learner(t_loss("mse"))$phash)
})

test_that("the hash covers the fields that define the learning problem", {
  d = tt_data()
  d$y = rnorm(nrow(d))
  task = tt_task(d, target = "y", id = "t")

  expect_equal(task$hash, tt_task(d, target = "y", id = "t")$hash)
  expect_false(task$hash == tt_task(d, target = "y", id = "t", output_dim = function(task) 5L)$hash)
  expect_false(task$hash == tt_task(d, target = "y", id = "t",
    prediction_encoder = function(task, predict_tensor, predict_type) list(response = 1))$hash)
  expect_equal(task$hash, task$clone(deep = TRUE)$hash)
})

test_that("train, predict and score work for a multi-label task", {
  task = tt_task_labels(60L)

  learner = tt_learner(tt_loss_bce())
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
  task = tt_task(d, target = "y")

  learner = tt_learner(t_loss("cross_entropy"))
  learner$predict_type = "prob"
  learner$train(task)
  pred = learner$predict(task)

  expect_factor(pred$response, levels = c("a", "b", "c"), len = task$nrow)
  expect_matrix(pred$prob, nrows = task$nrow, ncols = 3L)
  expect_equal(colnames(pred$prob), c("a", "b", "c"))
  expect_equal(unname(rowSums(pred$prob)), rep(1, task$nrow), tolerance = 1e-5)
})

test_that("train and predict work for a single numeric target", {
  d = tt_data(60L)
  d$y = d$x1 + rnorm(nrow(d))
  task = tt_task(d, target = "y")

  learner = tt_learner(t_loss("mse"))
  learner$train(task)
  pred = learner$predict(task)

  expect_numeric(pred$response, len = task$nrow)
  expect_numeric(pred$truth, len = task$nrow)
})

test_that("resampling works", {
  d = tt_data(60L)
  d$y = d$x1 + rnorm(nrow(d))
  task = tt_task(d, target = "y")
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
  task = tt_task(d, target = "y")
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
  autoenc = function(data, x) x[[1L]]
  task = tt_task(d, id = "ae", output_dim = function(task) 3L,
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
    target_batchgetter = autoenc)$.getbatch(1:4)
  expect_equal(as.matrix(batch$y), as.matrix(batch$x$x))

  learner = tt_learner(t_loss("mse"), epochs = 5L, target_batchgetter = autoenc)
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
  # nothing about the problem is specified by such a task either
  expect_error(get_target_batchgetter(task), "it is the learner that decides")
  expect_error(output_dim_for(task), "has no `output_dim`")
  expect_error(encode_prediction(task, torch_randn(2, 2), "response"), "has no `prediction_encoder`")

  task = tt_task(d, id = "unsup", output_dim = function(task) 2L,
    prediction_encoder = function(task, predict_tensor, predict_type) {
      list(response = as.matrix(predict_tensor$cpu()))
    })
  expect_equal(output_dim_for(task), 2L)
  # the batches carry no target
  expect_names(names(task_dataset(task, list(x = ingress_num()))$.getbatch(1:4)),
    permutation.of = c("x", ".index"))

  # the loss of a target-less task takes the network output alone
  loss = TorchLoss$new(nn_module("spread",
    initialize = function() NULL,
    forward = function(input) input$pow(2)$mean()), id = "spread")
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
  task = tt_task(d, id = "unsup", output_dim = function(task) 2L,
    prediction_encoder = function(task, predict_tensor, predict_type) {
      list(response = as.matrix(predict_tensor$cpu()))
    })
  loss = TorchLoss$new(nn_module("spread",
    initialize = function() NULL,
    forward = function(input) input$pow(2)$mean()), id = "spread")
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

test_that("an obs_loss declares what it asks for", {
  # the same arguments as the scoring function, and they add the same properties
  measure = msr_torch("a", function(truth, response) 1,
    obs_loss = function(task, prediction) rep(1, length(prediction$row_ids)))
  expect_set_equal(measure$properties, c("obs_loss", "requires_task"))

  # `Measure$obs_loss()` is never given a train_set, so asking for one cannot work
  expect_error(
    msr_torch("b", function(truth, response) 1, obs_loss = function(truth, train_set) 1),
    "arguments of `obs_loss`", fixed = TRUE
  )

  # two measures differing only in their obs_loss are not the same measure
  expect_false(
    msr_torch("c", function(truth, response) 1, obs_loss = function(truth) 1)$hash ==
      msr_torch("c", function(truth, response) 1, obs_loss = function(truth) 2)$hash
  )
})

test_that("the default measure does not fix an optimization direction", {
  d = tt_data(40L)
  d$y = rnorm(nrow(d))
  # a measure to be MAXIMIZED, which is what the default measure of a task may well be
  acc = msr_torch("acc", function(truth, response) mean(abs(truth - response) < 1),
    minimize = FALSE, range = c(0, 1))
  task = tt_task(d, target = "y", id = "t", default_measure = acc)

  # the direction belongs to the task's measure, which is not known when this one is constructed,
  # so it says so rather than guessing -- mlr3 refuses to tune with an NA direction
  expect_true(is.na(msr("torch.default")$minimize))
  expect_equal(msr("torch.default", minimize = FALSE, range = c(0, 1))$minimize, FALSE)

  # scoring works whatever the direction, because scoring does not consult it
  learner = tt_learner(t_loss("mse"))
  learner$train(task)
  expect_number(learner$predict(task)$score(msr("torch.default"), task = task), lower = 0, upper = 1)

  # ... but a stated direction that contradicts the task is an error, not a flipped ranking
  expect_error(
    learner$predict(task)$score(msr("torch.default", minimize = TRUE), task = task),
    "minimize = TRUE, but the default measure 'acc'"
  )
})

test_that("a task cannot be built or mutated into an inconsistent state", {
  d = tt_data(20L)
  d$y = rnorm(nrow(d))

  # duplicated targets used to give a task whose `$ncol` and `output_dim` counted them twice, and
  # only `$truth()` eventually failed, from inside the backend
  expect_error(tt_task(d, target = c("y", "y")), "Contains duplicated values")

  # the fields that define the learning problem are part of `$hash`, so a task that changed one
  # would no longer be the task a cached result was computed for
  task = tt_task(d, target = "y")
  expect_error(task$prediction_encoder <- 42, "read-only")
  expect_error(task$default_measure <- 42, "read-only")
  expect_function(task$prediction_encoder)
})

test_that("as_task_torch is the identity on a TaskTorch", {
  task = tt_task_labels(10L)
  expect_identical(as_task_torch(task), task)
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
  expect_number(pred$score(weighted, task = task), lower = 0, upper = 1)

  # mlr3 refuses a measure that does not declare them rather than ignoring them silently
  plain = msr_torch("ham", function(truth, response) mean(as.matrix(truth) != response),
    range = c(0, 1))
  expect_error(pred$score(plain, task = task), "does not support weights")

  # they are subset and combined with the observations they belong to
  pred$filter(task$row_ids[c(2L, 4L)])
  expect_numeric(pred$data$weights, len = 2L)
  rr = resample(task, learner, rsmp("cv", folds = 2L))
  expect_numeric(rr$prediction()$data$weights, len = task$nrow)
  expect_number(rr$aggregate(weighted), lower = 0, upper = 1)
})

test_that("a prediction never carries a predict type its learner does not have", {
  d = tt_data(30L)
  d$a = d$x1 > 0
  d$b = d$x2 > 0
  # an encoder that returns `prob` whatever it was asked for
  task = tt_task(d, target = c("a", "b"), id = "pt", output_dim = function(task) 2L,
    prediction_encoder = function(task, predict_tensor, predict_type) {
      prob = as.matrix(with_no_grad(nnf_sigmoid(predict_tensor))$to(device = "cpu"))
      colnames(prob) = task$target_names
      list(response = prob > 0.5, prob = prob)
    })

  learner = tt_learner(tt_loss_bce())
  learner$train(task)
  expect_equal(learner$predict(task)$predict_types, "response")

  # ... which is what lets it combine with the empty prediction of a failed fold
  expect_data_table(
    as.data.table(as_prediction(c(create_empty_prediction_data(task, learner),
      learner$predict(task)$data))),
    nrows = task$nrow
  )

  learner$predict_type = "prob"
  expect_set_equal(learner$predict(task)$predict_types, c("response", "prob"))
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

test_that("the graph language works", {
  d = tt_data(60L)
  d$y = d$x1 + rnorm(nrow(d))
  task = tt_task(d, target = "y")

  graph = po("torch_ingress_num") %>>%
    nn("linear_1", out_features = 10L) %>>%
    nn("relu_1") %>>%
    nn("head") %>>%
    po("torch_loss", t_loss("mse")) %>>%
    po("torch_optimizer", "adam") %>>%
    po("torch_model", batch_size = 16L, epochs = 3L, target_batchgetter = tt_bg)

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

test_that("predicting on zero rows gives an empty prediction", {
  task = tt_task_labels(40L, id = "t")
  learner = tt_learner(tt_loss_bce())
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

test_that("list valued predictions survive a resample round trip", {
  d = tt_data(60L)
  d$y = d$x1 + rnorm(nrow(d))
  task = tt_task(d, target = "y",
    prediction_encoder = function(task, predict_tensor, predict_type) {
      v = as.numeric(predict_tensor$cpu())
      # the predictions of two observations do not have the same length, so the only thing that can
      # hold them is a list
      list(response = lapply(seq_along(v), function(i) rep(v[i], 1L + i %% 2L)))
    })

  rr = resample(task, tt_learner(t_loss("mse")), rsmp("cv", folds = 3L))
  pred = rr$prediction()
  expect_list(pred$response, types = "numeric", len = task$nrow)
  expect_set_equal(lengths(pred$response), c(1L, 2L))

  # a list is one column of the table, not one column per observation
  tab = as.data.table(pred)
  expect_data_table(tab, nrows = task$nrow)
  expect_true(is.list(tab$response))
})

test_that("missing predictions are found whatever their storage", {
  pdata = function(response) {
    structure(list(row_ids = 1:3, response = response),
      class = c("PredictionDataTorch", "PredictionData"))
  }
  a = array(1, dim = c(3L, 2L, 2L))
  a[2L, 2L, 1L] = NA
  # `is.na()` of an array is an array of the same shape, so indexing the row ids with it used to
  # pick the wrong observation, or none at all
  expect_equal(is_missing_prediction_data(pdata(a)), 2L)
  expect_equal(is_missing_prediction_data(pdata(list(1, c(NA, 2), 3))), 2L)
  expect_equal(is_missing_prediction_data(pdata(matrix(c(1, NA, 3, 4, 5, 6), nrow = 3L))), 2L)
  expect_equal(is_missing_prediction_data(pdata(c(1, NA, 3))), 2L)
  expect_equal(is_missing_prediction_data(pdata(as_lazy_tensor(torch_randn(3L, 2L)))), integer(0))
})

test_that("a learner for a different task type is rejected", {
  d = tt_data()
  d$y = d$x1 + rnorm(nrow(d))
  task = tt_task(d, target = "y")
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
  task = tt_task(d, target = "y", id = "lt",
    output_dim = function(task) 3L,
    prediction_encoder = function(task, predict_tensor, predict_type) {
      list(response = as.matrix(predict_tensor$cpu()))
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

test_that("a task can predict standard errors", {
  d = tt_data(60L)
  d$y = d$x1 + rnorm(nrow(d))
  # a heteroscedastic network: one unit for the mean, one for the log standard deviation
  task = tt_task(d, target = "y", id = "se", output_dim = function(task) 2L,
    prediction_encoder = function(task, predict_tensor, predict_type) {
      out = as.matrix(with_no_grad(predict_tensor)$cpu())
      list(response = out[, 1L], se = if (predict_type == "se") exp(out[, 2L]))
    })

  expect_true("se" %chin% names(mlr_reflections$learner_predict_types$torch))
  # the two units mean different things, so the loss is a Gaussian negative log likelihood over
  # both of them rather than a distance to the target
  nll = nn_module("nn_test_gaussian_nll",
    initialize = function() NULL,
    forward = function(input, target) {
      mu = input[, 1L]
      log_sd = input[, 2L]
      torch_mean(log_sd + (target$squeeze(2L) - mu)^2 / (2 * torch_exp(2 * log_sd)))
    }
  )
  learner = tt_learner(TorchLoss$new(nll, task_types = "torch", id = "gaussian_nll"),
    predict_type = "se")
  learner$train(task)
  pred = learner$predict(task)

  expect_set_equal(pred$predict_types, c("response", "se"))
  expect_numeric(pred$se, len = task$nrow, lower = 0)
  expect_numeric(pred$response, len = task$nrow)

  # a measure can ask for it, and it survives a resample round trip
  measure = msr_torch("nll", predict_type = "se", range = c(-Inf, Inf),
    function(truth, response, se) mean(log(se) + (truth - response)^2 / (2 * se^2)))
  expect_number(pred$score(measure))
  rr = resample(task, learner, rsmp("cv", folds = 2L))
  expect_numeric(rr$prediction()$se, len = task$nrow, lower = 0)
  expect_number(rr$aggregate(measure))

  expect_true("se" %chin% names(as.data.table(pred)))
})
