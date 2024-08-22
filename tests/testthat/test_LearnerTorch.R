test_that("Basic checks", {
  learner = lrn("classif.torch_featureless", callbacks = "history")
  expect_learner(learner)
})

test_that("deep cloning", {
  learner = lrn("classif.torch_featureless", callbacks = "history")
  learner$param_set$set_values(epochs = 1, batch_size = 1)
  task = tsk("iris")
  learner$train(task)
  learner$state$train_task = NULL

  learner_cloned = learner$clone(deep = TRUE)
  expect_deep_clone(learner, learner_cloned)

  network = learner$network
  network_cloned = learner_cloned$network
  expect_true(torch_equal(network$weights, network_cloned$weights))
  network$weights$requires_grad_(FALSE)
  network$weights[1] = network$weights[1] + 1
  expect_false(torch_equal(network$weights, network_cloned$weights))

  expect_identical(
    get_private(learner)$.loss$generator,
    get_private(learner_cloned)$.loss$generator
  )

  expect_identical(
    get_private(learner)$.loss$generator,
    get_private(learner_cloned)$.loss$generator
  )

  expect_true(all(pmap_lgl(list(
      get_private(learner)$.callbacks$generator,
      get_private(learner_cloned)$.callbacks$generator),
    identical)))
})


test_that("Correct error when using problematic measures", {
  learner = lrn("classif.torch_featureless", epochs = 1, batch_size = 16, measures_train = msr("classif.bbrier"))
  task = tsk("german_credit")
  expect_error(learner$train(task), "Change the predict type or select other measures")
})

test_that("Correct error when providing measure with id 'epoch'", {
  m = msr("classif.acc")
  m$id = "epoch"
  expect_error(lrn("classif.torch_featureless", measures_train = m), "must not have id")
})

test_that("Basic tests: Classification", {
  learner = LearnerTorchTest1$new(task_type = "classif")
  expect_class(learner, c("LearnerTorchTest1", "LearnerTorch", "Learner"))
  expect_equal(learner$id, "classif.test1")
  expect_equal(learner$label, "Test1 Learner")
  expect_set_equal(learner$feature_types, c("numeric", "integer"))
  expect_set_equal(learner$properties, c("multiclass", "twoclass", "marshal", "validation", "internal_tuning"))

  # default predict types are correct
  expect_set_equal(learner$predict_types, c("response", "prob"))

  expect_subset(c("torch", "mlr3torch"), learner$packages)

  data = data.frame(x1 = 1:10, x2 = runif(10), y = 1:10)

  task = as_task_classif(data, target = "y", id = "hallo")

  learner$param_set$values$epochs = 0
  learner$param_set$values$batch_size = 1

  learner$train(task)
  expect_class(learner$network, "nn_module")
})

test_that("Basic tests: Regression", {
  learner = LearnerTorchTest1$new(task_type = "regr")
  expect_class(learner, c("LearnerTorchTest1", "LearnerTorch", "Learner"))
  expect_equal(learner$id, "regr.test1")
  expect_equal(learner$label, "Test1 Learner")
  expect_set_equal(learner$feature_types, c("numeric", "integer"))
  expect_set_equal(learner$properties, c("marshal", "validation", "internal_tuning"))

  # default predict types are correct
  expect_set_equal(learner$predict_types, "response")

  expect_subset(c("torch", "mlr3torch"), learner$packages)

  data = data.frame(x1 = 1:10, x2 = runif(10), y = 1:10)

  task = as_task_regr(data, target = "y", id = "hallo")

  learner$param_set$values$epochs = 0
  learner$param_set$values$batch_size = 1

  learner$train(task)
  expect_class(learner$network, "nn_module")
})


test_that("Param Set for optimizer and loss are correctly created", {
  opt = t_opt("sgd")
  loss = t_loss("cross_entropy")
  cb = t_clbk("checkpoint")
  # loss$param_set$subset(c("weight", "ignore_index"))
  learner = lrn("classif.torch_featureless", optimizer = opt, loss = loss, callbacks = cb)
  expect_subset(paste0("opt.", opt$param_set$ids()), learner$param_set$ids())
  expect_subset(paste0("loss.", loss$param_set$ids()), learner$param_set$ids())
  expect_subset(paste0("cb.checkpoint.", cb$param_set$ids()), learner$param_set$ids())
})


test_that("Parameters cannot start with {loss, opt, cb}.", {
  helper = function(param_set) {
    R6Class("LearnerTorchTest",
      inherit = LearnerTorch,
      public = list(
        initialize = function(optimizer = t_opt("adagrad"), loss = t_loss("cross_entropy")) {
          super$initialize(
            task_type = "classif",
            id = "classif.test1",
            label = "Test1 Classifier",
            feature_types = c("numeric", "integer"),
            param_set = param_set,
            properties = c("multiclass", "twoclass"),
            predict_types = "response",
            optimizer = optimizer,
            loss = loss,
            man = "mlr3torch::mlr_learners.test1"
          )
        }
      )
    )$new()
  }

  expect_error(helper(ps(loss.weight = p_dbl())), "are reserved for")
  expect_error(helper(ps(opt.weight = p_dbl())), "are reserved for")
  expect_error(helper(ps(cb.weight = p_dbl())), "are reserved for")
})

test_that("ParamSet reference identities are preserved after a deep clone", {
  # Explanation: When we call $get_optimizer() or $get_loss(), the paramset of private$.optimizer and private$.loss
  # are used. The paramsets in the ParamSetCollection must therefore point to these ParamSets so that values set
  # by calling learner$param_set$set_values() also have an effect during training.
  # This is solved by setting the private$.param_set to NULL in the deep clone, so that it is reconstructed correctly
  # afterwards

  learner = LearnerTorchTest1$new(task_type = "classif")
  learner1 = learner$clone(deep = TRUE)

  learner1$param_set$set_values(opt.lr = 9.99)
  expect_true(get_private(learner1)$.optimizer$param_set$values$lr == 9.99)
  learner1$param_set$set_values(loss.weight = 0.11)
  expect_true(get_private(learner1)$.loss$param_set$values$weight == 0.11)
})

test_that("Learner inherits packages from optimizer, loss, and callbacks", {
  tcb = torch_callback("custom", packages = "utils")
  opt = t_opt("adam")
  opt$packages = "base"
  loss = t_loss("cross_entropy")
  loss$packages = "stats"
  learner = LearnerTorchFeatureless$new(
    task_type = "classif",
    callbacks = list(tcb),
    loss = loss,
    optimizer = opt
  )
  expect_subset(c("utils", "stats", "base"), learner$packages)
})

test_that("Train-predict loop is reproducible when setting a seed", {
  learner1 = lrn("classif.torch_featureless", batch_size = 16, epochs = 1, predict_type = "prob", shuffle = TRUE,
    seed = 1)
  task = tsk("iris")

  learner2 = lrn("classif.torch_featureless", batch_size = 16, epochs = 1, predict_type = "prob", shuffle = TRUE,
    seed = 1)

  learner1$train(task)
  learner2$train(task)

  p1 = learner1$predict(task)
  p2 = learner2$predict(task)

  expect_identical(p1$prob, p2$prob)
})

test_that("the state of a trained network contains what it should", {
  task = tsk("mtcars")$select("am")

  learner = lrn("regr.torch_featureless", epochs = 0, batch_size = 10,
    callbacks = t_clbk("history", id = "history1"),
    optimizer = t_opt("sgd", lr = 1),
    loss = t_loss("l1")
  )
  learner$train(task)
  expect_permutation(
    names(learner$model),
    c("seed", "network", "optimizer", "loss_fn", "task_col_info", "callbacks", "epochs", "internal_valid_scores")
  )
  expect_true(is.integer(learner$model$seed))
  expect_class(learner$model$network, "nn_module")
  expect_list(learner$model$loss_fn)
  expect_list(learner$model$optimizer)
  expect_list(learner$model$callbacks)
  expect_equal(names(learner$model$callbacks), "history1")
  expect_true(is.integer(learner$model$seed))
  expect_permutation(learner$model$task_col_info$id, c("mpg", "am"))
  expect_permutation(colnames(learner$model$task_col_info), c("id", "type", "levels"))
})

test_that("train parameters do what they should: classification and regression", {
  withr::local_seed(1L)
  callback = torch_callback(id = "internals",
    on_begin = function() {
      # rename to avoid deleting the ctx after finishing the training
      self$ctx1 = self$ctx
    },
    load_state_dict = function(state_dict) {
      NULL
    },
    state_dict = function() {
      list(ctx = self$ctx1, num_threads = torch_get_num_threads())
    }
  )

  f = function(task_type, measure_ids) {
    task = switch(task_type, regr = tsk("mtcars"), classif = tsk("iris"))
    epochs = sample(10:12, 1)
    batch_size = sample(2:3, 1)
    shuffle = sample(c(TRUE, FALSE), 1)
    num_threads = if (running_on_mac()) 1L else sample(2, 1)
    drop_last = sample(c(TRUE, FALSE), 1)
    seed = sample.int(10, 1)
    measures_train = msrs(paste0(measure_ids[sample(c(TRUE, FALSE), 2, replace = FALSE)]))
    measures_valid = msrs(paste0(measure_ids[sample(c(TRUE, FALSE), 2, replace = FALSE)]))

    learner = lrn(paste0(task_type, ".torch_featureless"),
      epochs = epochs,
      batch_size = batch_size,
      callbacks = list(callback, t_clbk("history")),
      shuffle = shuffle,
      num_threads = num_threads,
      drop_last = drop_last,
      seed = seed,
      measures_train = measures_train,
      measures_valid = measures_valid,
      predict_type = switch(task_type, classif = "prob", regr = "response"),
      device = "cpu",
      validate = "predefined"
    )

    # first we test everything with validation
    learner$validate = 0.3
    learner$train(task)

    internals = learner$model$callbacks$internals
    ctx = internals$ctx

    if (!running_on_mac()) {
      expect_equal(num_threads, internals$num_threads)
    }
    expect_equal(ctx$loader_train$batch_size, batch_size)
    expect_equal(ctx$loader_valid$batch_size, batch_size)
    expect_equal(ctx$total_epochs, epochs)
    expect_equal(ctx$network$parameters[[1]]$device$type, "cpu")

    if (shuffle) {
      expect_class(ctx$loader_train$sampler, "utils_sampler_random")
    } else {
      expect_class(ctx$loader_train$sampler, "utils_sampler_sequential")
    }
    expect_class(ctx$loader_valid$sampler, "utils_sampler_sequential")
    if (drop_last) {
      expect_true(ctx$loader_train$drop_last)
    } else {
      expect_false(ctx$loader_train$drop_last)
    }

    expect_false(ctx$loader_valid$drop_last)

    expect_equal(nrow(learner$model$callbacks$history$valid), epochs)
    expect_equal(nrow(learner$model$callbacks$history$train), epochs)
    expect_permutation(c("epoch", ids(measures_train)), colnames(learner$model$callbacks$history$train))
    expect_permutation(c("epoch", ids(measures_valid)), colnames(learner$model$callbacks$history$valid))

    # now without validation
    learner$validate = NULL

    learner$state = NULL
    learner$param_set$values$measures_valid = list()
    learner$train(task)

    expect_equal(nrow(learner$model$callbacks$history$valid), 0)


    learner$validate = 0.2
    learner$state = NULL
    learner$param_set$set_values(
      device = "meta",
      epochs = 0,
      measures_valid = list()
    )

    # FIXME: extend this to all dataloader parameters

    learner$train(task)
    ctx = learner$model$callbacks$internals$ctx
    loader_train_iter = dataloader_make_iter(ctx$loader_train)
    loader_valid = dataloader_make_iter(ctx$loader_valid)

    # now we also test that the device placement works
    # expect_equal(learner$network$parameters[[1]]$device$type, "meta")
  }

  f("classif", c("classif.acc", "classif.ce"))
  f("regr", c("regr.mse", "regr.mae"))
})

test_that("predict types work during training and prediction", {
  # Here we check that when setting the predict type to "prob", they are available during training
  # (and hence also for validation)
  task = tsk("iris")
  learner = lrn("classif.torch_featureless", epochs = 1, batch_size = 16, predict_type = "prob",
    measures_train = msr("classif.mbrier"), callbacks = t_clbk("history"))
  learner$train(task)
  expect_true(!is.na(learner$model$callbacks$history$train[1, "classif.mbrier"][[1L]]))

  pred = learner$predict(task)
  expect_true(is.matrix(pred$prob))
  expect_true(is.factor(pred$response))
  expect_equal(levels(pred$response), task$class_names)
  expect_permutation(colnames(pred$prob), task$class_names)
  expect_prediction_classif(pred)

  learner$predict_type = "response"
  pred = learner$predict(task)
  expect_true(is.null(pred$prob))
  expect_true(is.factor(pred$response))
  expect_equal(levels(pred$response), task$class_names)
  expect_prediction_classif(pred)
})

test_that("predict parameters do what they should: classification and regression", {
  # Currently available predict parameters:
  # * batch_size
  # * device
  # * num_threads
  # * seed (already checked somewhere else)

  callback = torch_callback(id = "internals",
    on_begin = function() {
      # Rename so it won't get deleted after training finishes
      self$ctx1 = self$ctx
    },
    state_dict = function() {
      list(ctx = self$ctx1, num_threads = torch_get_num_threads())
    },
    load_state_dict = function(state_dict) NULL
  )

  f = function(task_type) {
    num_threads = if (running_on_mac()) 1L else sample(2, 1)
    batch_size = sample(16, 1)
    learner = lrn(paste0(task_type, ".torch_featureless"), epochs = 1, callbacks = callback,
      num_threads = num_threads,
      batch_size = batch_size,
      shuffle = TRUE
    )
    task = switch(task_type, regr = tsk("mtcars"), classif = tsk("iris"))
    learner$train(task)
    internals = learner$model$callbacks$internals
    ctx = internals$ctx
    if (!running_on_mac()) {
      expect_equal(num_threads, internals$num_threads)
    }

    learner$param_set$set_values(device = "meta")

    dl = get_private(learner)$.dataloader_predict(task, learner$param_set$values)
    expect_equal(dl$batch_size, batch_size)
    expect_class(dl$sampler, "utils_sampler_sequential")
  }

  f("regr")
  f("classif")
})

test_that("quick accessors work", {
  task = tsk("mtcars")
  learner = lrn("regr.torch_featureless", epochs = 1, batch_size = 1, callbacks = "history")
  expect_true(is.null(learner$network))
  learner$train(task)
  expect_class(learner$network, "nn_module")
})

test_that("Train-Predict works", {
  learner = lrn("classif.torch_featureless", epochs = 1, device = "cpu", batch_size = 16)
  task = tsk("iris")
  split = partition(task)
  learner$train(task, row_ids = split$train)
  pred = learner$predict(task, row_ids = split$test)

  expect_prediction_classif(pred)

  expect_equal(task$truth(split$test), pred$truth)
})

# This should not really be needed but see:
# https://github.com/mlr-org/mlr3/issues/947
test_that("resample() works", {
  learner = lrn("regr.torch_featureless", epochs = 1, batch_size = 50)
  task = tsk("mtcars")
  resampling = rsmp("holdout")
  rr = resample(task, learner, resampling)
  expect_r6(rr, "ResampleResult")
})

test_that("marshaling", {
  task = tsk("mtcars")$filter(1:5)
  learner = lrn("regr.mlp", batch_size = 150, epochs = 1, device = "cpu", encapsulate = c(train = "callr"),
    neurons = 20
  )
  learner$train(task)
  expect_false(learner$marshaled)
  learner$marshal()$unmarshal()
  expect_prediction(learner$predict(task))
})

test_that("callr encapsulation and marshaling", {
  skip_if_not_installed("callr")
  task = tsk("mtcars")$filter(1:5)
  learner = lrn("regr.mlp", batch_size = 150, epochs = 1, device = "cpu", encapsulate = c(train = "callr"),
    neurons = 20
  )
  learner$train(task)
  expect_prediction(learner$predict(task))
})

test_that("future and marshaling", {
  skip_if_not_installed("future")
  task = tsk("mtcars")$filter(1:5)
  learner = lrn("regr.mlp", batch_size = 150, epochs = 1, device = "cpu",
    neurons = 20
  )
  rr = with_future(future::multisession, {
    resample(task, learner, rsmp("holdout"))
  })
  expect_class(rr, "ResampleResult")
})

test_that("Input verification works during `$train()` (train-predict shapes work together)", {
  task = nano_mnist()

  task_invalid = po("trafo_resize", size = c(10, 10), stages = "train") $train(list(task))[[1L]]
  task_valid = po("trafo_resize", size = c(10, 10), stages = "both") $train(list(task))[[1L]]

  learner = lrn("classif.torch_featureless",
    batch_size = 1L, epochs = 0L
  )

  # fallback learner cannot help in this case!
  learner$fallback = lrn("classif.featureless")
  rr_faulty = resample(task_invalid, learner, rsmp("holdout"))
  expect_true(nrow(rr_faulty$errors) == 1L)
  rr1 = resample(task, learner, rsmp("holdout"))
  expect_true(nrow(rr1$errors) == 0L)

  task_unknown = po("trafo_resize", size = c(10, 10), stages = "train") $train(list(nano_dogs_vs_cats()))[[1L]]
  rr2 = resample(task_unknown, learner, rsmp("holdout"))
  expect_true(nrow(rr2$errors) == 0L)
})

test_that("col_info is propertly subset when comparing task validity during predict", {
  task = tsk("iris")$select("Sepal.Length")
  learner = classif_mlp2()
  learner$train(task)
  learner$model$task_col_info
  expect_permutation(learner$model$task_col_info$id, c("Sepal.Length", "Species"))

  task2 = tsk("iris")
  task2$cbind(data.frame(x = rnorm(150)))$select("Sepal.Length")

  expect_class(learner$predict(task), "PredictionClassif")
})

test_that("deep clone works", {
  l1 = lrn("classif.mlp")
  l2 = l1$clone(deep = TRUE)
  expect_deep_clone(l1, l2)
})

test_that("param set is read-only", {
  learner = lrn("classif.mlp")
  expect_error({learner$param_set = ps()}, "read-only")
})

test_that("(p)hash", {
  expect_eq_phash = function(x, y) expect_equal(x$phash, y$phash)
  expect_eq_hash = function(x, y) expect_equal(x$hash, y$hash)
  expect_ne_phash = function(x, y) expect_false(x$phash == y$phash)
  expect_ne_hash = function(x, y) expect_false(x$hash == y$hash)

  expect_ne_hash(lrn("regr.mlp"), lrn("classif.mlp"))
  expect_ne_hash(lrn("regr.mlp"), lrn("regr.mlp", epochs = 1))
  expect_eq_phash(lrn("regr.mlp"), lrn("regr.mlp", epochs = 1))

  expect_ne_hash(lrn("regr.mlp"), lrn("regr.mlp", optimizer = "sgd"))
  expect_ne_hash(lrn("regr.mlp", loss = "mse"), lrn("regr.mlp", loss = "l1"))
  expect_ne_hash(lrn("regr.mlp"), lrn("regr.mlp", callbacks = t_clbk("history")))
})

test_that("eval_freq works", {
  learner = lrn("regr.torch_featureless", epochs = 10, batch_size = 50, eval_freq = 4, callbacks = "history",
    measures_train = msrs("regr.mse"), measures_valid = msrs("regr.mse"), validate = 0.3)
  task = tsk("mtcars")
  learner$train(task)
  expect_equal(learner$model$callbacks$history$valid$epoch, c(4, 8, 10))
  expect_equal(learner$model$callbacks$history$train$epoch, c(4, 8, 10))
})

test_that("early stopping works", {
  learner = lrn("classif.torch_featureless", epochs = 10, batch_size = 50, eval_freq = 3, callbacks = "history",
    measures_valid = msr("classif.ce"), validate = 0.3, patience = 2, min_delta = 2)
  task = tsk("iris")

  learner$train(task)
  # the first evaluation can do no comparison, i.e. the second eval with no improvement is the third epoch
  expect_equal(learner$internal_tuned_values, list(epochs = 9))

  # in this scenario early stopping should definitely not trigger yet
  learner$param_set$set_values(
    min_delta = 0, patience = 5, opt.lr = 0.01, eval_freq = 1
  )
  learner$train(task)
  expect_equal(learner$internal_tuned_values, list(epochs = 10))
})

test_that("validation works", {
  task = tsk("mtcars")
  task$internal_valid_task = task

  learner = lrn("regr.torch_featureless", epochs = 20, batch_size = 150, eval_freq = 3,
    measures_valid = msr("regr.mse"), validate = "predefined", seed = 1, opt.lr = 1)

  learner$train(task)
  expect_list(learner$internal_valid_scores, "numeric", len = 1L)
  expect_equal(names(learner$internal_valid_scores), "regr.mse")
  expect_true(abs(var(task$truth()) - learner$internal_valid_scores[[1L]]) < 2)
})

test_that("validation measure must specify minimize when early stopping", {
  measure = msr("regr.mse")
  measure$minimize = NA
  learner = lrn("regr.torch_featureless", epochs = 1, batch_size = 50,
    measures_valid = measure, validate = 0.2, opt.lr = 1, patience = 1)

  expect_error(learner$train(tsk("mtcars")), "NA")
})

test_that("internal tuning", {
  skip_if_not_installed("mlr3tuning")
  task = tsk("iris")
  lgr::get_logger("bbotk")$set_threshold("warn")
  learner = lrn("classif.torch_featureless",
    epochs = to_tune(upper = 10, internal = TRUE),
    batch_size = to_tune(10, 20), eval_freq = 3, measures_valid = msr("classif.ce"),
    validate = 0.3, patience = 2, min_delta = 2
  )

  ti = mlr3tuning::tune(
    tuner = mlr3tuning::tnr("grid_search", batch_size = 2),
    learner = learner,
    task = tsk("iris"),
    resampling = rsmp("holdout"),
    term_evals = 2
  )
  expect_equal(
    ti$archive$data$internal_tuned_values, replicate(list(list(epochs = 9L)), n = 2L)
  )
  expect_equal(ti$result_learner_param_vals$epochs, 9L)
})


test_that("param_set source works", {
  l = R6Class("LearnerTorchTest",
    inherit = LearnerTorch,
    public = list(
      initialize = function() {
        private$.ps1 = ps(a = p_int(tags = c("train", "required")))
        super$initialize(
          task_type = "regr", id = "test", properties = c(),
          param_set = alist(private$.ps1),
          feature_types = "numeric",
          label = "Test Learner",
          callbacks = t_clbk("checkpoint"),
          man = "some_man"
        )
      }
    ),
    private = list(.ps1 = NULL)
  )$new()
  l$param_set$set_values(
    a = 7,
    epochs = 8,
    opt.lr = 9,
    loss.reduction = "mean",
    cb.checkpoint.freq = 3
  )
  expect_equal(l$param_set$values$a, 7)
  expect_equal(get_private(l)$.ps1$values$a, 7)
  expect_equal(l$param_set$values$epochs, 8)
  expect_equal(get_private(l)$.param_set_torch$values$epochs, 8)
  expect_equal(l$param_set$values$opt.lr, 9)
  expect_equal(get_private(l)$.optimizer$param_set$values$lr, 9)
  expect_equal(l$param_set$values$loss.reduction, "mean")
  expect_equal(get_private(l)$.loss$param_set$values$reduction, "mean")
  expect_equal(l$param_set$values$cb.checkpoint.freq, 3)
  expect_equal(get_private(l)$.callbacks[[1L]]$param_set$values$freq, 3)

  l1 = l$clone(deep = TRUE)

  expect_deep_clone(l, l1)
  l1$param_set$set_values(
    a = 17,
    epochs = 18,
    opt.lr = 19,
    loss.reduction = "sum",
    cb.checkpoint.freq = 13
  )

  expect_equal(l1$param_set$values$a, 17)
  expect_equal(get_private(l1)$.ps1$values$a, 17)
  expect_equal(l1$param_set$values$epochs, 18)
  expect_equal(get_private(l1)$.param_set_torch$values$epochs, 18)
  expect_equal(l1$param_set$values$opt.lr, 19)
  expect_equal(get_private(l1)$.optimizer$param_set$values$lr, 19)
  expect_equal(l1$param_set$values$loss.reduction, "sum")
  expect_equal(get_private(l1)$.loss$param_set$values$reduction, "sum")
  expect_equal(l1$param_set$values$cb.checkpoint.freq, 13)
  expect_equal(get_private(l1)$.callbacks[[1]]$param_set$values$freq, 13)
})

test_that("one feature works", {
  task = tsk("mtcars")$select("am")
  learner = lrn("regr.mlp", epochs = 1L, batch_size = 150)
  learner$train(task)
  expect_class(learner, "Learner")
  pred = learner$predict(task)
  expect_class(pred, "Prediction")
})

test_that("param_set alist must refer to self, private or super", {
  LearnerTest = R6Class("LearnerTest", inherit = LearnerTorch,
    public = list(
      initialize = function(loss = NULL, optimizer = NULL, callbacks = list(), param_set) {
        self$ps1 = ps(a = p_int(tags = "train"))
        private$ps2 = ps(b = p_int(tags = "train"))
        super$initialize(
          "regr",
          id = "test",
          label = "Test",
          loss = loss,
          callbacks = callbacks,
          optimizer = optimizer,
          param_set = param_set,
          properties = c(),
          feature_types = "integer",
          man = NA
        )
      },
      ps1 = NULL
    ),
    private = list(
      ps2 = NULL
    )
  )

  learner = LearnerTest$new(param_set = alist(self$ps1, private$ps2))
  expect_subset(c("a", "b"), learner$param_set$ids())
  expect_error(LearnerTest$new(param_set = alist(ps(c = p_int(tags = "train")))))
})

test_that("configure loss, optimizer and callbacks after construction", {
  learner = lrn("classif.torch_model",
    loss = LossNone(),
    optimizer = OptimizerNone(),
    callbacks = CallbacksNone()
  )

  expect_true(is.null(learner$loss))
  expect_true(is.null(learner$optimizer))
  expect_true(is.null(learner$callbacks))

  expect_false(any(grepl("^loss\\.", learner$param_set$ids())))
  expect_error({learner$loss = t_loss("mse")}) # nolint

  loss = t_loss("cross_entropy")
  loss$packages = c(loss$packages, "utils")
  learner$loss = loss
  expect_true("loss.reduction" %in% learner$param_set$ids())
  expect_true("utils" %in% learner$packages)
  expect_false(any(grepl("^opt\\.", learner$param_set$ids())))
  expect_class(learner$loss, "TorchLoss")
  learner$param_set$set_values(loss.reduction = "sum")
  expect_equal(learner$param_set$values$loss.reduction, "sum")
  expect_equal(learner$loss$param_set$values$reduction, "sum")

  expect_error({learner$optimizer = 1L}) # nolint
  optimizer = t_opt("adam")
  optimizer$packages = c(optimizer$packages, "stats")
  learner$optimizer = optimizer
  expect_true("stats" %in% optimizer$packages)
  expect_true("opt.amsgrad" %in% learner$param_set$ids())
  expect_class(learner$optimizer, "TorchOptimizer")
  learner$param_set$set_values(opt.lr = 2)
  expect_equal(learner$param_set$values$opt.lr, 2)
  expect_equal(learner$optimizer$param_set$values$lr, 2)

  expect_false(any(grepl("^cb\\.", learner$param_set$ids())))
  expect_error({learner$callbacks = list(1L)}) # nolint
  callback = t_clbk("checkpoint")
  callback$packages = c(callback$packages, "R6")
  learner$callbacks = list(callback)
  expect_true("cb.checkpoint.freq" %in% learner$param_set$ids())
  expect_list(learner$callbacks, "TorchCallback")
  learner$param_set$set_values(cb.checkpoint.freq = 100)
  expect_equal(learner$param_set$values$cb.checkpoint.freq, 100)
  expect_equal(learner$callbacks[[1]]$param_set$values$freq, 100)

  learner$param_set$set_values(
    loss.reduction = "mean",
    opt.lr = 123,
    cb.checkpoint.freq = 456
  )
  learner1 = learner$clone(deep = TRUE)
  expect_deep_clone(learner, learner1)
  expect_equal(learner1$param_set$values$loss.reduction, "mean")
  expect_equal(learner1$param_set$values$opt.lr, 123)
  expect_equal(learner1$param_set$values$cb.checkpoint.freq, 456)
})

test_that("dataset works", {
  task = tsk("iris")
  learner = lrn("classif.mlp", device = "meta", batch_size = 10,
    epochs = 1L)
  ds = learner$dataset(task)
  batch = ds$.getbatch(1:2)
  expect_equal(batch$x$torch_ingress_num.input$device$type, "meta")
  expect_equal(batch$x$torch_ingress_num.input$shape, c(2, 4))
  expect_equal(batch$y$device$type, "meta")
  expect_equal(batch$y$shape, 2)
  expect_equal(batch$.index$device$type, "meta")
  expect_equal(batch$.index$shape, 2)
  skip_if(torch::cuda_is_available())
  learner$param_set$set_values(device = "auto")
  ds = learner$dataset(task)
  batch = ds$.getbatch(1:2)
  expect_equal(batch$x$torch_ingress_num.input$device$type, "cpu")
})

test_that("error when dataloaders have length 0", {
  learner = lrn("regr.torch_featureless", epochs = 1L, batch_size = 100, drop_last = TRUE)
  task = tsk("mtcars")
  expect_error({learner$train(task)}, "has length 0") # nolint
})

test_that("can set seed to NULL", {
  task = tsk("iris")
  l = lrn("classif.torch_featureless", epochs = 1, batch_size = 150, seed = NULL)
  l$train(task)
  l$predict(task)
  expect_true(is.null(l$model$seed))
})
