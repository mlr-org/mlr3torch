simple_graph = function() {
  po("torch_ingress_num") %>>%
    nn("linear", out_features = 5) %>>%
    nn("relu") %>>%
    nn("head") %>>%
    po("torch_loss", "cross_entropy") %>>%
    po("torch_optimizer", "adam", lr = 0.1)
}

test_that("classification: train and predict", {
  task = tsk("iris")
  learner = as_learner_torch(simple_graph(), task_type = "classif", epochs = 1, batch_size = 32)

  expect_class(learner, "LearnerTorch")
  expect_equal(learner$id, "classif.graph")
  expect_equal(learner$task_type, "classif")
  expect_equal(learner$feature_types, c("integer", "numeric"))

  learner$train(task)
  expect_class(learner$network, c("nn_graph", "nn_module"))
  expect_class(learner$model, "learner_torch_model")

  pred = learner$predict(task)
  expect_class(pred, "PredictionClassif")
  expect_equal(pred$row_ids, task$row_ids)
})

test_that("regression: train and predict", {
  task = tsk("mtcars")
  graph = po("torch_ingress_num") %>>% nn("head") %>>% po("torch_optimizer", "sgd", lr = 0.001)
  learner = as_learner_torch(graph, task_type = "regr", epochs = 1, batch_size = 16)

  expect_class(learner, "LearnerTorch")
  expect_equal(learner$id, "regr.graph")
  # no `po("torch_loss")` in the graph, so the LearnerTorch default is used
  expect_equal(learner$loss$id, "mse")

  learner$train(task)
  expect_class(learner$network, c("nn_graph", "nn_module"))
  pred = learner$predict(task)
  expect_class(pred, "PredictionRegr")
})

test_that("the network and the dataset match the ingress of the graph", {
  task = tsk("iris")
  learner = as_learner_torch(simple_graph(), task_type = "classif", epochs = 1, batch_size = 32)
  learner$train(task)

  expect_equal(names(learner$network$shapes_in), "torch_ingress_num.input")
  ds = learner$dataset(task)
  batch = ds$.getbatch(1:2)
  expect_equal(names(batch$x), "torch_ingress_num.input")
  out = with_no_grad(invoke(learner$network, .args = batch$x))
  expect_equal(out$shape, c(2, 3))
})

test_that("predictions are identical to the GraphLearner route", {
  withr::local_seed(1)
  task = tsk("iris")
  graph = simple_graph()

  learner = as_learner_torch(graph, task_type = "classif", epochs = 2, batch_size = 32, seed = 1)
  learner$predict_type = "prob"
  learner$train(task)

  glrn = as_learner(graph %>>% po("torch_model_classif", epochs = 2, batch_size = 32, seed = 1))
  glrn$predict_type = "prob"
  glrn$train(task)

  expect_equal(learner$predict(task)$prob, glrn$predict(task)$prob)
})

test_that("hyperparameters of the graph are exposed and used", {
  learner = as_learner_torch(simple_graph(), task_type = "classif", epochs = 1, batch_size = 32)

  expect_subset(c("linear.out_features", "linear.bias", "head.bias"), learner$param_set$ids())
  expect_equal(learner$param_set$values$linear.out_features, 5)
  # the parameters of the consumed configuration operators move to the LearnerTorch names
  expect_subset(c("opt.lr", "loss.reduction"), learner$param_set$ids())
  expect_false(any(startsWith(learner$param_set$ids(), "torch_")))
  expect_equal(learner$param_set$values$opt.lr, 0.1)

  learner$param_set$set_values(linear.out_features = 7)
  learner$train(tsk("iris"))
  expect_equal(learner$network$graph$pipeops$linear$module$out_features, 7)
})

test_that("loss, optimizer and callbacks are taken from the graph", {
  graph = simple_graph() %>>% po("torch_callbacks", t_clbk("checkpoint", freq = 2))
  learner = as_learner_torch(graph, task_type = "classif", epochs = 1, batch_size = 32)

  expect_equal(learner$loss$id, "cross_entropy")
  expect_equal(learner$optimizer$id, "adam")
  expect_equal(ids(learner$callbacks), "checkpoint")
  expect_subset("cb.checkpoint.freq", learner$param_set$ids())
  expect_equal(learner$param_set$values$cb.checkpoint.freq, 2)
})

test_that("constructor arguments take precedence over the graph", {
  learner = as_learner_torch(simple_graph(), task_type = "classif", optimizer = t_opt("sgd", lr = 0.5),
    loss = t_loss("cross_entropy", reduction = "sum"), id = "my_learner")

  expect_equal(learner$id, "my_learner")
  expect_equal(learner$optimizer$id, "sgd")
  expect_equal(learner$param_set$values$opt.lr, 0.5)
  expect_equal(learner$param_set$values$loss.reduction, "sum")
})

test_that("a terminal PipeOpTorchModel determines the task type and the parameter values", {
  graph = simple_graph() %>>% po("torch_model_classif", epochs = 3, batch_size = 8)
  learner = as_learner_torch(graph)

  expect_equal(learner$task_type, "classif")
  expect_equal(learner$param_set$values$epochs, 3)
  expect_equal(learner$param_set$values$batch_size, 8)
  # `epochs` must not be exposed a second time under the PipeOp's name
  expect_false(any(startsWith(learner$param_set$ids(), "torch_model_classif.")))

  learner$train(tsk("iris"))
  expect_class(learner$network, "nn_graph")
})

test_that("PipeOp and GraphLearner methods", {
  learner = as_learner_torch(po("torch_ingress_num") %>>% nn("head"), task_type = "regr",
    epochs = 1, batch_size = 16)
  expect_class(learner, "LearnerTorch")

  glrn = as_learner(simple_graph() %>>% po("torch_model_classif", epochs = 1, batch_size = 32))
  learner2 = as_learner_torch(glrn)
  expect_class(learner2, "LearnerTorch")
  expect_equal(learner2$task_type, "classif")
  expect_equal(learner2$param_set$values$epochs, 1)
})

test_that("multiple ingress operators", {
  task = tsk("iris")
  graph = gunion(list(
    po("select", id = "select_sepal", selector = selector_grep("^Sepal")) %>>%
      po("torch_ingress_num", id = "sepal"),
    po("select", id = "select_petal", selector = selector_grep("^Petal")) %>>%
      po("torch_ingress_num", id = "petal")
  )) %>>% nn("merge_cat") %>>% nn("head")

  learner = as_learner_torch(graph, task_type = "classif", epochs = 1, batch_size = 32)
  # the task is preprocessed before it reaches the ingress, so the learner cannot restrict its
  # feature types to the ones the ingress accepts
  expect_set_equal(learner$feature_types, unname(mlr_reflections$task_feature_types))

  learner$train(task)
  expect_set_equal(names(learner$network$shapes_in), c("sepal.input", "petal.input"))
  expect_set_equal(names(learner$dataset(task)$.getbatch(1)$x), c("sepal.input", "petal.input"))
  expect_class(learner$predict(task), "PredictionClassif")
})

test_that("lazy tensor ingress", {
  task = tsk("lazy_iris")
  graph = po("torch_ingress_ltnsr") %>>% nn("head")
  learner = as_learner_torch(graph, task_type = "classif", epochs = 1, batch_size = 32)

  expect_equal(learner$feature_types, "lazy_tensor")
  learner$train(task)
  expect_equal(learner$network$shapes_in[["torch_ingress_ltnsr.input"]], c(NA, 4))
  expect_class(learner$predict(task), "PredictionClassif")
})

test_that("graphs that cannot be converted give an informative error", {
  expect_error(as_learner_torch(po("pca"), task_type = "classif"), "no PipeOpTorchIngress")
  expect_error(as_learner_torch(simple_graph()), "Cannot infer the task type")
  expect_error(
    as_learner_torch(simple_graph() %>>% po("torch_model_classif"), task_type = "regr"),
    "but the graph contains PipeOp 'torch_model_classif' with task type 'classif'"
  )
  expect_error(
    as_learner_torch(po("torch_ingress_num") %>>% nn("head") %>>%
      po("torch_loss", "cross_entropy", id = "loss1") %>>%
      po("torch_loss", "mse", id = "loss2"), task_type = "classif"),
    "more than one PipeOpTorchLoss"
  )
  expect_error(
    as_learner_torch(gunion(list(po("torch_ingress_num"), po("torch_ingress_num", id = "second"))),
      task_type = "classif"),
    "2 output channels"
  )
  learner = as_learner_torch(po("torch_ingress_num") %>>% nn("head"), task_type = "classif")
  expect_error(learner$dataset(tsk("iris"), train = FALSE), "must be trained before")
})

batch_mean = function(ds, n) {
  as.numeric(ds$.getbatch(seq_len(n))$x[[1L]]$mean())
}

test_that("a stateful operator before the ingress predicts with the training state", {
  task = tsk("iris")
  graph = po("scale") %>>% po("torch_ingress_num") %>>% nn("head")
  learner = as_learner_torch(graph, task_type = "classif", epochs = 1, batch_size = 32)
  learner$train(task)

  expect_names(names(learner$model$ingress$states), permutation.of = c("scale", "torch_ingress_num"))
  expect_numeric(learner$model$ingress$states$scale$center, len = 4L)

  # only the first class, whose features are far from the mean of the whole task
  setosa = task$clone()$filter(1:50)
  # the prediction phase standardizes with the statistics of the training task, ...
  expect_true(abs(batch_mean(learner$dataset(setosa), 50L)) > 0.3)
  # ... whereas the training phase would fit them on the task it is given
  expect_equal(batch_mean(learner$dataset(setosa, train = TRUE), 50L), 0, tolerance = 1e-5)

  expect_class(learner$predict(setosa), "PredictionClassif")
  expect_class(learner$predict_newdata(task$data(1:5)), "PredictionClassif")
})

test_that("predictions match the GraphLearner route with a stateful operator in the graph", {
  withr::local_seed(1)
  task = tsk("iris")
  graph = po("scale") %>>% po("torch_ingress_num") %>>% nn("linear", out_features = 5) %>>%
    nn("relu") %>>% nn("head") %>>% po("torch_loss", "cross_entropy") %>>%
    po("torch_optimizer", "adam", lr = 0.1)
  ids = partition(task)

  learner = as_learner_torch(graph, task_type = "classif", epochs = 2, batch_size = 32, seed = 1)
  learner$predict_type = "prob"
  learner$train(task, ids$train)

  glrn = as_learner(graph %>>% po("torch_model_classif", epochs = 2, batch_size = 32, seed = 1))
  glrn$predict_type = "prob"
  glrn$train(task, ids$train)

  expect_equal(learner$predict(task, ids$test)$prob, glrn$predict(task, ids$test)$prob)
})

test_that("the stages of a preprocessing operator are respected", {
  task = tsk("lazy_iris")
  # doubling the features is easy to recognize in the tensors and does not change their shape
  po_augment = po("preproc_torch", id = "double",
    fn = crate(function(x) x * 2), stages = "train")
  graph = po_augment %>>% po("torch_ingress_ltnsr", shape = c(NA, 4)) %>>% nn("head")
  learner = as_learner_torch(graph, task_type = "classif", epochs = 1, batch_size = 32)
  learner$train(task)

  raw = batch_mean(as_learner_torch(po("torch_ingress_ltnsr") %>>% nn("head"), task_type = "classif",
    epochs = 1, batch_size = 32)$dataset(task), 10L)
  # `stages = "train"` means the transformation belongs to training only
  expect_equal(batch_mean(learner$dataset(task, train = TRUE), 10L), 2 * raw)
  expect_equal(batch_mean(learner$dataset(task), 10L), raw)
})

test_that("marshaling and serialization with a stateful operator before the ingress", {
  task = tsk("iris")
  graph = po("scale") %>>% po("torch_ingress_num") %>>% nn("head")
  learner = as_learner_torch(graph, task_type = "classif", epochs = 1, batch_size = 32)
  learner$train(task)
  pred = learner$predict(task)
  center = learner$model$ingress$states$scale$center

  path = tempfile()
  on.exit(unlink(path), add = TRUE)
  learner$marshal()
  expect_true(learner$marshaled)
  saveRDS(learner, path)
  learner2 = readRDS(path)$unmarshal()

  expect_equal(learner2$model$ingress$states$scale$center, center)
  expect_equal(learner2$predict(task)$response, pred$response)
})

test_that("the internal validation task is transformed like prediction data", {
  task = tsk("iris")
  graph = po("scale") %>>% po("torch_ingress_num") %>>% nn("head")
  learner = as_learner_torch(graph, task_type = "classif", epochs = 3, batch_size = 32,
    measures_valid = msr("classif.ce"), patience = 2)
  set_validate(learner, 0.3)
  learner$train(task)

  expect_number(learner$internal_valid_scores$classif.ce)
  expect_number(learner$internal_tuned_values$epochs)
  # the state was fitted on the training part only, not on the whole task
  expect_false(isTRUE(all.equal(unname(learner$model$ingress$states$scale$center),
    unname(colMeans(as.matrix(task$data(cols = task$feature_names)))))))
})

test_that("resampling works with a stateful operator before the ingress", {
  graph = po("scale") %>>% po("torch_ingress_num") %>>% nn("head")
  learner = as_learner_torch(graph, task_type = "classif", epochs = 1, batch_size = 32)
  rr = resample(tsk("iris"), learner, rsmp("cv", folds = 2L), store_models = TRUE)
  expect_double(rr$aggregate())
  # every fold fitted its own state
  centers = map(rr$learners, function(l) l$model$ingress$states$scale$center)
  expect_false(isTRUE(all.equal(centers[[1L]], centers[[2L]])))
})

test_that("autotest", {
  learner = as_learner_torch(simple_graph(), task_type = "classif", epochs = 1, batch_size = 32)
  expect_learner_torch(learner, task = tsk("iris"))
})
