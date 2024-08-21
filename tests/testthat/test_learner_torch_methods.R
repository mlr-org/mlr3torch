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
    target_batchgetter = crate(function(data, device) {
      torch_tensor(data = as.integer(data[[1]]), dtype = torch_long(), device = device)
    }, .parent = topenv()),
    device = "cpu"
  )

  dataloader1 = dataloader(
    dataset = dataset1,
    batch_size = 3L,
    drop_last = FALSE,
    shuffle = TRUE
  )
  pred = torch_network_predict(net1, dataloader1)
  expect_error(torch_network_predict(net2, dataloader1))


  ingress2 = list(
    x1 = TorchIngressToken("Sepal.Length", batchgetter_num, c(NA, 1L))
  )

  dataset2 = task_dataset(
    task,
    feature_ingress_tokens = ingress2,
    target_batchgetter = crate(function(data, device) {
      torch_tensor(data = as.integer(data[[1]]), dtype = torch_long(), device = device)
    }, .parent = topenv()),
    device = "cpu"
  )

  dataloader2 = dataloader(
    dataset = dataset2,
    batch_size = 3L,
    shuffle = TRUE
  )

  net3 = nn_linear(1, 3)

  weight_before = torch_clone(net3$weight)

  pred = torch_network_predict(net3, dataloader2)

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

  expect_data_table(learner$model$callbacks$history$train, nrows = 2)
  expect_equal(colnames(learner$model$callbacks$history$train), c("epoch", "classif.acc"))

  expect_true(nrow(learner$model$callbacks$history$valid) == 0)
  expect_equal(colnames(learner$model$callbacks$history$valid), "epoch")

  learner = lrn("classif.torch_featureless", epochs = 2, batch_size = 1, measures_train = msrs(c("classif.acc")),
    measures_valid = msr("classif.bacc"), callbacks = t_clbk("history"), validate = "predefined"
  )

  learner$train(task)

  expect_true(nrow(learner$model$callbacks$history$train) == 2)
  expect_true(nrow(learner$model$callbacks$history$valid) == 2)
})

test_that("learner_torch_predict works", {
  task = tsk("iris")
  learner = lrn("classif.mlp", batch_size = 16, epochs = 1, device = "cpu")
  dl = get_private(learner)$.dataloader(task, learner$param_set$values)

  network = get_private(learner)$.network(task, learner$param_set$values)

  pred = torch_network_predict(network, dl)

  expect_class(pred, "torch_tensor")
  expect_true(ncol(pred) == length(task$class_names))
  expect_true(nrow(pred) == task$nrow)

})

test_that("encode_prediction_default works", {
  task = tsk("iris")

  # classif
  pt = torch_rand(task$nrow, length(task$class_names))
  pt = pt / torch_sum(pt, 2L)$reshape(c(150, 1))

  p1 = encode_prediction_default(pt, "response", task)
  p2 = encode_prediction_default(pt, "prob", task)

  pd1 = as_prediction_data(p1, task)
  pd2 = as_prediction_data(p2, task)

  expect_identical(pd1$response, pd2$response)

  expect_identical(p1$response, p2$response)
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
  dl = get_private(learner)$.dataloader_predict(task, learner$param_set$values)
  expect_false(dl$drop_last)
  expect_class(dl$batch_sampler$sampler, "utils_sampler_sequential")
})
