test_that("torch_network_predict works", {
  task = tsk("iris")

  net1 = nn_module(
    initialize = function() {
      self$linear1 = nn_linear(1, 3)
      self$linear2 = nn_linear(1, 3)
    },
    forward = function(x1, x2) {
      self$linear1(x1) + self$linear2(x2)
    }
  )()

  net2 = nn_module(
    initialize = function() {
      self$linear1 = nn_linear(1, 3)
      self$linear2 = nn_linear(1, 3)
    },
    forward = function(a1, a2) {
      self$linear1(a1) + self$linear2(a2)
    }
  )()

  ingress1 = list(
    x1 = TorchIngressToken("Sepal.Length", batchgetter_num, c(NA, 1L)),
    x2 = TorchIngressToken("Sepal.Width", batchgetter_num, c(NA, 1L))
  )

  dataset1 = task_dataset(
    task,
    feature_ingress_tokens = ingress1,
    target_batchgetter = crate(function(data) {
      torch_tensor(data = as.integer(data[[1]]), dtype = torch_long())
    }, .parent = topenv())
  )

  dataloader1 = dataloader(
    dataset = dataset1,
    batch_size = 3L,
    drop_last = FALSE,
    shuffle = TRUE
  )
  pred = torch_network_predict(net1, dataloader1, device = "cpu")
  expect_error(torch_network_predict(net2, dataloader1, device = "cpu"))


  ingress2 = list(
    x1 = TorchIngressToken("Sepal.Length", batchgetter_num, c(NA, 1L))
  )

  dataset2 = task_dataset(
    task,
    feature_ingress_tokens = ingress2,
    target_batchgetter = crate(function(data) {
      torch_tensor(data = as.integer(data[[1]]), dtype = torch_long())
    }, .parent = topenv())
  )

  dataloader2 = dataloader(
    dataset = dataset2,
    batch_size = 3L,
    shuffle = TRUE
  )

  net3 = nn_linear(1, 3)

  weight_before = torch_clone(net3$weight)

  pred = torch_network_predict(net3, dataloader2, device = "cpu")

  expect_true(torch_equal(weight_before, net3$weight))

})

test_that("Validation Task is respected", {
  task = tsk("iris")
  task$internal_valid_task = task$clone(deep = TRUE)$filter(1:10)
  task$row_roles$use = 1:10

  learner = lrn("classif.torch_featureless", epochs = 2, batch_size = 1, measures_train = msrs(c("classif.acc")),
    callbacks = t_clbk("history"), validate = "predefined"
  )
  learner$train(task)

  expect_data_table(learner$model$callbacks$history, nrows = 2)
  expect_equal(colnames(learner$model$callbacks$history), c("epoch", "train.classif.acc"))

  learner = lrn("classif.torch_featureless", epochs = 2, batch_size = 1, measures_train = msrs(c("classif.acc")),
    measures_valid = msr("classif.bacc"), callbacks = t_clbk("history"), validate = "predefined"
  )

  learner$train(task)

  expect_data_table(learner$model$callbacks$history, nrows = 2)
  expect_equal(colnames(learner$model$callbacks$history), c("epoch", "train.classif.acc", "valid.classif.bacc"))
})

test_that("learner_torch_predict works", {
  check = function(task, ncol) {
    learner = lrn("classif.mlp", batch_size = 16, epochs = 1, device = "cpu")
    dl = get_private(learner)$.dataloader(
      get_private(learner)$.dataset(task, learner$param_set$values), learner$param_set$values)

    network = get_private(learner)$.network(task, learner$param_set$values)

    pred = torch_network_predict(network, dl, device = "cpu")

    expect_class(pred, "torch_tensor")
    expect_equal(ncol(pred), ncol)
    expect_equal(nrow(pred), task$nrow)
  }
  check(tsk("iris"), 3)
  check(tsk("sonar"), 1)
  check(tsk("mtcars"), 1)
})

test_that("encode_prediction works", {
  check_classif = function(task, ncol) {
    pt = torch_rand(task$nrow, output_dim_for(task))
    pt = pt / torch_sum(pt, 2L)$reshape(c(task$nrow, 1))

    p1 = encode_prediction(task, pt, "response")
    p2 = encode_prediction(task, pt, "prob")

    pd1 = as_prediction_data(p1, task)
    pd2 = as_prediction_data(p2, task)

    expect_identical(pd1$response, pd2$response)
    expect_identical(p1$response, p2$response)
  }
  check_classif(tsk("iris"), 3)
  check_classif(tsk("sonar"), 1)

  check_regr = function(task, ncol) {
    pt = torch_rand(task$nrow, 1)
    p = encode_prediction(task, pt, "response")
    expect_equal(as.numeric(p$response), as.numeric(pt))
  }
  check_regr(tsk("mtcars"), 1)
})

test_that("the built-in encodings accept one head and reject more", {
  regr = tsk("mtcars")
  pt = torch_rand(regr$nrow, 1)
  expect_equal(encode_prediction(regr, list(pt), "response"), encode_prediction(regr, pt, "response"))
  expect_error(encode_prediction(regr, list(mu = pt, sigma = pt), "response"),
    "returned 2 tensors, but the prediction encoding for task type 'regr' expects a single one")

  classif = tsk("iris")
  pt = torch_rand(classif$nrow, output_dim_for(classif))
  expect_equal(encode_prediction(classif, list(pt), "prob"), encode_prediction(classif, pt, "prob"))
  expect_error(encode_prediction(classif, list(pt, pt), "prob"),
    "returned 2 tensors, but the prediction encoding for task type 'classif' expects a single one")
})

test_that("torch_network_predict concatenates the heads of a network that returns a list", {
  task = tsk("iris")
  learner = lrn("classif.mlp", batch_size = 16, epochs = 1, device = "cpu")
  dl = get_private(learner)$.dataloader_predict(
    get_private(learner)$.dataset(task, learner$param_set$values), learner$param_set$values)

  # the two heads have different widths, so a mix-up between them cannot go unnoticed
  network = nn_module(
    initialize = function() NULL,
    forward = function(x) list(identity = x, total = x$sum(dim = 2L, keepdim = TRUE))
  )()

  pred = torch_network_predict(network, dl, device = "cpu")

  expect_list(pred, types = "torch_tensor", len = 2L)
  expect_equal(names(pred), c("identity", "total"))
  expect_equal(dim(pred$identity), c(task$nrow, length(task$feature_names)))
  expect_equal(dim(pred$total), c(task$nrow, 1L))
  expect_true(torch_allclose(pred$total, pred$identity$sum(dim = 2L, keepdim = TRUE)))
})

test_that("a network with two heads can be trained, scored and predicted", {
  two_heads = nn_module("nn_two_heads",
    initialize = function(d_in) {
      self$mu = nn_linear(d_in, 1)
      self$log_sigma = nn_linear(d_in, 1)
    },
    forward = function(x) list(mu = self$mu(x), log_sigma = self$log_sigma(x))
  )

  LearnerTwoHeads = R6Class("LearnerTwoHeads",
    inherit = LearnerTorch,
    public = list(
      initialize = function() {
        super$initialize(task_type = "regr", id = "regr.two_heads", label = "Two Heads",
          param_set = ps(), feature_types = c("numeric", "integer"), man = "mlr3torch::two_heads")
      }
    ),
    private = list(
      .network = function(task, param_vals) two_heads(length(task$feature_names)),
      .ingress_tokens = function(task, param_vals) {
        list(x = ingress_num(shape = c(NA, length(task$feature_names))))
      },
      # the configured loss is applied to a single tensor, so it is wrapped to see only the mean
      .loss_fn = function(task, param_vals) {
        nn_module("nn_mu_loss",
          initialize = function(loss) self$loss = loss,
          forward = function(input, target) self$loss(input$mu, target)
        )(super$.loss_fn(task, param_vals))
      },
      .encode_prediction = function(predict_tensor, task) {
        list(response = as.numeric(predict_tensor$mu + torch_exp(predict_tensor$log_sigma)))
      }
    )
  )

  # the features are scaled, otherwise the exponentiated head overflows
  task = po("scale")$train(list(tsk("mtcars")))[[1L]]
  learner = LearnerTwoHeads$new()
  learner$param_set$set_values(epochs = 1L, batch_size = 16L, measures_valid = msr("regr.mse"))
  set_validate(learner, 0.3)
  learner$train(task)

  # the validation scores go through the same encoding as the prediction
  expect_list(learner$internal_valid_scores, types = "numeric", len = 1L)
  expect_false(is.na(learner$internal_valid_scores$regr.mse))

  pred = learner$predict(task)
  expect_class(pred, "PredictionRegr")

  # both heads made it into the response, in the right order
  x = torch_tensor(as.matrix(task$data(cols = task$feature_names)), dtype = torch_float())
  out = with_no_grad(learner$model$network(x))
  expect_equal(pred$response, as.numeric(out$mu + torch_exp(out$log_sigma)), tolerance = 1e-5)
})

test_that("Train and predict are reproducible and seeds work as expected", {
  # the with_torch_settings() functions is separately tested as well
  task = tsk("iris")

  # First we check that seed = "random" (the default) works
  learner = lrn("classif.torch_featureless", batch_size = 150, epochs = 2, predict_type = "prob")
  learner$train(task)
  p1 = learner$predict(task, row_ids = 1)
  expect_integer(learner$state$model$seed)
  learner$param_set$set_values(seed = learner$state$model$seed)
  learner$train(task)
  p2 = learner$predict(task, row_ids = 1)
  expect_equal(p1$prob, p2$prob)

  # Now we check that the seed we set is also used
  learner$param_set$set_values(seed = 1)
  learner$train(task)
  expect_equal(learner$model$seed, 1)
  p3 = learner$predict(task, row_ids = 1)
  learner$train(task)
  p4 = learner$predict(task, row_ids = 1)
  expect_equal(p1$prob, p2$prob)

  # This is just a sanity check that not simply everything is always the same
  expect_true(grepl(all.equal(p1$prob, p3$prob), pattern = "Mean relative"))
})

test_that("learner_torch_dataloader_predict works", {
  learner = lrn("regr.torch_featureless", batch_size = 15, drop_last = TRUE, device = "cpu",
    epochs = 1, shuffle = TRUE
  )
  task = tsk("iris")
  dl = get_private(learner)$.dataloader_predict(
    get_private(learner)$.dataset(task, learner$param_set$values), learner$param_set$values)
  expect_false(dl$drop_last)
  expect_class(dl$batch_sampler$sampler, "utils_sampler_sequential")
})

test_that("correct prob predictions for classification", {
  learner = lrn("classif.mlp", batch_size = 150, epochs = 0, device = "cpu", neurons = integer(),
    predict_type = "prob")

  rr_multi = resample(tsk("iris"), learner, rsmp("cv", folds = 2))
  pred_multi = rr_multi$prediction()
  prob_multi = pred_multi$prob
  expect_prediction_classif(pred_multi)
  # for each row, give the column name with the highest probability
  response_multi = apply(prob_multi, 1, function(x) colnames(prob_multi)[which.max(x)])
  expect_equal(as.character(pred_multi$response), response_multi)
  # ensure that the response prediction is the one with highest probability


  rr_binary = resample(tsk("sonar"), learner, rsmp("cv", folds = 2))
  pred_binary = rr_binary$prediction()
  prob_binary = pred_binary$prob
  expect_prediction_classif(pred_binary)
  # for each row, give the column name with the highest probability
  response_binary = apply(prob_binary, 1, function(x) colnames(prob_binary)[which.max(x)])
  expect_equal(as.character(pred_binary$response), response_binary)
  # ensure that the response prediction is the one with highest probability
  # for each row, give the column name with the highest probability
  response = apply(prob_binary, 1, function(x) colnames(prob_binary)[which.max(x)])
  expect_equal(as.character(pred_binary$response), response)
  # ensure that the response prediction is the one with highest probability
})
