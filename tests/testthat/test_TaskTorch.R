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
  expect_error(encode_prediction(bare, torch_randn(2, 1), "response"), "has no `default_encoder`")
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
  encoder = function(task, network_output, predict_type) list(response = rep(1, nrow(network_output)))
  task = as_task_torch(d, target = "y", id = "t", default_encoder = encoder)

  expect_equal(encode_prediction(task, torch_randn(3, 7), "response")$response, rep(1, 3))
  # the raw network output is handed over, so a multi-head network is the encoder's business
  multi = as_task_torch(d, target = "y", id = "t",
    default_encoder = function(task, network_output, predict_type) {
      list(response = as.numeric(network_output[[2L]]$cpu()))
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
    default_encoder = function(task, network_output, predict_type) list(response = 1))$hash)
  expect_equal(task$hash, task$clone(deep = TRUE)$hash)
})

test_that("train, predict and score work for a multi-label task", {
  task = tt_task_labels(60L)

  learner = tt_learner(tt_loss_bce(), predict_types = c("response", "prob"))
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

  learner = tt_learner(t_loss("cross_entropy"), predict_types = c("response", "prob"))
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

test_that("benchmark and tune work on a generic torch task", {
  d = tt_data(60L)
  d$y = d$x1 + rnorm(nrow(d))
  task = tt_task(d, target = "y", id = "t")
  measure = msr_torch("mse", function(truth, response) mean((truth - response)^2),
    range = c(0, Inf), minimize = TRUE)

  # `benchmark()` insists on distinct ids, so that its rows can be told apart afterwards
  short = tt_learner(t_loss("mse"), epochs = 1L)
  short$id = "short"
  long = tt_learner(t_loss("mse"), epochs = 3L)
  long$id = "long"
  bmr = benchmark(benchmark_grid(task, list(short, long), rsmp("cv", folds = 2L)))

  tab = bmr$aggregate(measure)
  expect_data_table(tab, nrows = 2L)
  expect_numeric(tab[[measure$id]], lower = 0, any.missing = FALSE)
  expect_set_equal(tab$learner_id, c("short", "long"))
  expect_equal(bmr$n_resample_results, 2L)
  # each row keeps its own predictions, which is what scoring per row relies on
  expect_equal(nrow(as.data.table(bmr$resample_result(1L)$prediction())), task$nrow)

  # tuning ranks the archive by the measure, which has to state a direction
  tuned = tt_learner(t_loss("mse"), epochs = to_tune(1L, 3L))
  ti = mlr3tuning::tune(mlr3tuning::tnr("grid_search", resolution = 3L), task, tuned,
    rsmp("holdout"), measure)
  expect_data_table(ti$archive$data, nrows = 3L)
  expect_number(ti$result_y, lower = 0)
  expect_true(ti$result$epochs %in% 1:3)
  # the winner is the smallest score, because the measure said so
  expect_equal(unname(ti$result_y), min(ti$archive$data[[measure$id]]))

  # ... and the task's default measure states none, so it cannot be tuned against
  expect_error(
    mlr3tuning::tune(mlr3tuning::tnr("grid_search", resolution = 2L), task, tuned,
      rsmp("holdout"), msr("torch.default")),
    "minimize` field set to NA"
  )
})

test_that("an AutoTuner over a generic torch task can be resampled", {
  d = tt_data(60L)
  d$y = d$x1 + rnorm(nrow(d))
  task = tt_task(d, target = "y", id = "t")
  measure = msr_torch("mse", function(truth, response) mean((truth - response)^2),
    range = c(0, Inf), minimize = TRUE)

  at = mlr3tuning::auto_tuner(
    tuner = mlr3tuning::tnr("grid_search", resolution = 2L),
    learner = tt_learner(t_loss("mse"), epochs = to_tune(1L, 2L)),
    resampling = rsmp("holdout"),
    measure = measure,
    term_evals = 2L
  )
  # nested resampling: the outer loop scores a learner that tunes itself on the inner one
  rr = resample(task, at, rsmp("cv", folds = 2L))
  expect_number(rr$aggregate(measure), lower = 0)
  expect_numeric(rr$prediction()$response, len = task$nrow)
})

test_that("validation and early stopping work", {
  d = tt_data(60L)
  d$y = d$x1 + rnorm(nrow(d))
  task = tt_task(d, target = "y")
  measure = msr_torch("mse", function(truth, response) mean((truth - response)^2),
    range = c(0, Inf), minimize = TRUE)

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
    default_encoder = function(task, network_output, predict_type) {
      response = as.matrix(network_output$cpu())
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
  expect_error(encode_prediction(task, torch_randn(2, 2), "response"), "has no `default_encoder`")

  task = tt_task(d, id = "unsup", output_dim = function(task) 2L,
    default_encoder = function(task, network_output, predict_type) {
      list(response = as.matrix(network_output$cpu()))
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
    default_encoder = function(task, network_output, predict_type) {
      list(response = as.matrix(network_output$cpu()))
    })
  loss = TorchLoss$new(nn_module("spread",
    initialize = function() NULL,
    forward = function(input) input$pow(2)$mean()), id = "spread")
  measure = msr_torch("spread", function(prediction) mean(prediction$response^2),
    range = c(0, Inf), minimize = TRUE)

  learner = tt_learner(loss, epochs = 10L, patience = 2L, measures_valid = measure)
  learner$validate = 0.3
  learner$train(task)
  expect_names(names(learner$internal_valid_scores), identical.to = "spread")

  learner = tt_learner(loss, tensor_dataset = TRUE)
  learner$train(task)
  expect_matrix(learner$predict(task)$response, nrows = task$nrow, ncols = 2L)
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
  expect_error(task$default_encoder <- 42, "read-only")
  expect_error(task$default_measure <- 42, "read-only")
  expect_function(task$default_encoder)
})

test_that("as_task_torch is the identity on a TaskTorch", {
  task = tt_task_labels(10L)
  expect_identical(as_task_torch(task), task)
})

test_that("as_task_torch says what to do with another Task", {
  # `as_data_backend()` has no method for a Task, and its dispatch error names nothing actionable
  expect_error(as_task_torch(tsk("mtcars"), target = "mpg"),
    "is a <TaskRegr> and cannot be converted into a TaskTorch", fixed = TRUE)

  # the route the message points at has to work
  source = tsk("mtcars")
  task = as_task_torch(source$data(), target = source$target_names, id = source$id)
  expect_class(task, "TaskTorch")
  expect_equal(task$target_names, source$target_names)
  expect_equal(task$nrow, source$nrow)
})

test_that("prob and se are opt-in for a torch learner", {
  # whether probabilities exist is decided by the task's `default_encoder`, which is unknown
  # when the learner is built, so claiming them by default let `predict_type = "prob"` through
  # only for the prediction to come back without any
  learner = tt_learner(t_loss("mse"))
  expect_equal(learner$predict_types, "response")
  expect_error({learner$predict_type = "prob"}, "does not support predict type 'prob'")
  expect_error({learner$predict_type = "se"}, "does not support predict type 'se'")

  expect_set_equal(tt_learner(t_loss("mse"), predict_types = c("response", "prob"))$predict_types,
    c("response", "prob"))
  # the route through a Graph has to offer it too, or a GraphLearner could never predict prob
  expect_set_equal(po("torch_model", predict_types = c("response", "prob"))$learner$predict_types,
    c("response", "prob"))
  expect_equal(po("torch_model")$learner$predict_types, "response")

  # classif and regr keep the defaults mlr3 expects of them
  expect_set_equal(lrn("classif.torch_featureless")$predict_types, c("response", "prob"))
  expect_equal(lrn("regr.torch_featureless")$predict_types, "response")
})

test_that("the prediction is whatever the encoder returned", {
  d = tt_data(30L)
  d$a = d$x1 > 0
  d$b = d$x2 > 0
  # an encoder that returns `prob` whatever it was asked for -- nothing filters it out, the encoder
  # is trusted to return what the `predict_type` it is handed asks for
  task = tt_task(d, target = c("a", "b"), id = "pt", output_dim = function(task) 2L,
    default_encoder = function(task, network_output, predict_type) {
      prob = as.matrix(with_no_grad(nnf_sigmoid(network_output))$to(device = "cpu"))
      colnames(prob) = task$target_names
      list(response = prob > 0.5, prob = prob)
    })

  learner = tt_learner(tt_loss_bce(), predict_types = c("response", "prob"))
  learner$train(task)
  expect_set_equal(learner$predict(task)$predict_types, c("response", "prob"))

  # the empty prediction of a failed fold asks the same encoder, so the two still combine
  expect_data_table(
    as.data.table(as_prediction(c(create_empty_prediction_data(task, learner),
      learner$predict(task)$data))),
    nrows = task$nrow
  )

  learner$predict_type = "prob"
  expect_set_equal(learner$predict(task)$predict_types, c("response", "prob"))
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

test_that("a task can predict standard errors", {
  d = tt_data(60L)
  d$y = d$x1 + rnorm(nrow(d))
  # a heteroscedastic network: one unit for the mean, one for the log standard deviation
  task = tt_task(d, target = "y", id = "se", output_dim = function(task) 2L,
    default_encoder = function(task, network_output, predict_type) {
      out = as.matrix(with_no_grad(network_output)$cpu())
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
    predict_types = c("response", "se"), predict_type = "se")
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
