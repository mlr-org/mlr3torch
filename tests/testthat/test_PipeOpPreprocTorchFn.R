test_that("PipeOpPreprocTorchTrafoFn autotest", {
  obj = po("trafo_fn", fn = function(x, a) x * a, a = 2)
  expect_class(obj, "PipeOpPreprocTorchTrafoFn")
  expect_equal(obj$param_set$values$stages, "both")
  expect_pipeop_torch_preprocess(obj, shapes_in = list(c(4, 3), c(2, 3, 5)), in_package = FALSE,
    deterministic = TRUE)
})

test_that("PipeOpPreprocTorchAugmentFn autotest", {
  obj = po("augment_fn", fn = function(x) x$flip(-1L))
  expect_class(obj, "PipeOpPreprocTorchAugmentFn")
  expect_equal(obj$param_set$values$stages, "train")
  expect_pipeop_torch_preprocess(obj, shapes_in = list(c(4, 3), c(2, 3, 5)), in_package = FALSE,
    deterministic = TRUE)
})

test_that("trafo_fn is applied when training and predicting, augment_fn only when training", {
  task = tsk("lazy_iris")$filter(1:3)
  x = as_array(materialize(task$data(cols = "x")$x, rbind = TRUE))

  apply_po = function(id, stage) {
    obj = po(id, fn = function(x) x * 2)
    out = obj$train(list(task))
    if (stage == "predict") out = obj$predict(list(task))
    as_array(materialize(out[[1L]]$data(cols = "x")$x, rbind = TRUE))
  }

  expect_equal(apply_po("trafo_fn", "train"), x * 2)
  expect_equal(apply_po("trafo_fn", "predict"), x * 2)
  expect_equal(apply_po("augment_fn", "train"), x * 2)
  expect_equal(apply_po("augment_fn", "predict"), x)
})

test_that("parameters are inferred from fn", {
  task = tsk("lazy_iris")$filter(1:3)
  x = as_array(materialize(task$data(cols = "x")$x, rbind = TRUE))

  obj = po("trafo_fn", fn = function(x, a, b = 1, ...) x * a + b)
  expect_set_equal(obj$param_set$ids(), c("a", "b", "stages", "affect_columns"))
  expect_true(all(map_lgl(obj$param_set$tags[c("a", "b")], function(tags) "train" %in% tags)))
  # an argument without a default must be set
  expect_true("required" %in% obj$param_set$tags$a)
  expect_false("required" %in% obj$param_set$tags$b)
  expect_error(obj$train(list(task)), "Missing required parameters")

  # an unset parameter falls back to the default of the argument
  obj$param_set$set_values(a = 3)
  out = obj$train(list(task))[[1L]]
  expect_equal(as_array(materialize(out$data(cols = "x")$x, rbind = TRUE)), x * 3 + 1, tolerance = 1e-6)

  # a function of the tensor alone has no further parameters
  expect_set_equal(po("trafo_fn", fn = function(x) x)$param_set$ids(), c("stages", "affect_columns"))
  expect_error(po("trafo_fn", fn = function() 1), "first argument")
})

test_that("a custom param_set can be passed", {
  fn = function(x, a) x * a
  obj = po("trafo_fn", fn = fn, param_set = ps(a = p_dbl(lower = 0, tags = c("train", "required"))), a = 3)
  expect_equal(obj$param_set$class[["a"]], "ParamDbl")
  expect_error(obj$param_set$set_values(a = -1), "not >=")

  # the parameters must be arguments of fn ...
  expect_error(po("trafo_fn", fn = fn, param_set = ps(b = p_dbl(tags = "train"))), "ids of param_set")
  # ... unless it has `...`
  expect_class(po("trafo_fn", fn = function(x, ...) x, param_set = ps(b = p_dbl(tags = "train"))),
    "PipeOpPreprocTorchTrafoFn")
  # and they must only be used during training
  expect_error(po("trafo_fn", fn = fn, param_set = ps(a = p_dbl(tags = "required"))), "tagged 'train'")
  expect_error(po("trafo_fn", fn = fn, param_set = ps(a = p_dbl(tags = c("train", "predict")))), "tagged 'train'")
})

test_that("shapes_out", {
  fn = function(x) torch_cat(list(x, x), dim = 2L)

  # inferred by default
  expect_equal(po("trafo_fn", fn = fn)$shapes_out(list(c(NA, 3L)), stage = "train"), list(c(NA, 6L)))
  # unknown
  expect_equal(po("trafo_fn", fn = fn, shapes_out = NULL)$shapes_out(list(c(NA, 3L)), stage = "train"), list(NULL))
  # custom function
  shapes_out = function(shapes_in, param_vals, task) list(c(NA, 2L * shapes_in[[1L]][2L]))
  expect_equal(po("trafo_fn", fn = fn, shapes_out = shapes_out)$shapes_out(list(c(NA, 3L)), stage = "train"),
    list(c(NA, 6L)))
  expect_error(po("trafo_fn", fn = fn, shapes_out = function(x) x), "shapes_in")
  obj = po("trafo_fn", fn = fn, shapes_out = function(shapes_in, param_vals, task) c(NA, 6L))
  expect_error(obj$shapes_out(list(c(NA, 3L)), stage = "train"), "return value of 'shapes_out'")

  # rowwise: the function is applied to the elements of the batch
  obj = po("trafo_fn", fn = function(x) x$unsqueeze(1L), rowwise = TRUE)
  expect_true(obj$rowwise)
  expect_equal(obj$shapes_out(list(c(NA, 3L)), stage = "train"), list(c(NA, 1L, 3L)))
  task = tsk("lazy_iris")$filter(1:2)
  out = obj$train(list(task))[[1L]]
  expect_equal(materialize(out$data(cols = "x")$x, rbind = TRUE)$shape, c(2L, 1L, 4L))
})

test_that("phash depends on fn, rowwise and shapes_out", {
  f1 = function(x) x * 2
  f2 = function(x) x * 3
  expect_equal(po("trafo_fn", fn = f1)$phash, po("trafo_fn", fn = f1)$phash)
  expect_false(po("trafo_fn", fn = f1)$phash == po("trafo_fn", fn = f2)$phash)
  expect_false(po("trafo_fn", fn = f1)$phash == po("trafo_fn", fn = f1, rowwise = TRUE)$phash)
  expect_false(po("trafo_fn", fn = f1)$phash == po("trafo_fn", fn = f1, shapes_out = NULL)$phash)
  expect_false(po("trafo_fn", fn = f1)$phash == po("augment_fn", fn = f1, id = "trafo_fn")$phash)
})

test_that("trafo_fn and augment_fn can be used in a learner", {
  task = tsk("lazy_iris")
  graph = po("augment_fn", fn = function(x, sd) x + torch_randn_like(x) * sd, sd = 0.01) %>>%
    po("trafo_fn", fn = function(x) torch_cat(list(x, x^2), dim = 2L)) %>>%
    po("torch_ingress_ltnsr") %>>%
    nn("head") %>>%
    po("torch_loss", t_loss("cross_entropy")) %>>%
    po("torch_optimizer", t_opt("adam")) %>>%
    po("torch_model_classif", epochs = 1L, batch_size = 50L)
  learner = as_learner(graph)
  learner$train(task)
  # the network is built for the inferred shape of the transformed features
  network = learner$model$torch_model_classif$model$network
  expect_equal(network$parameters[[1L]]$shape, c(3L, 8L))
  expect_class(learner$predict(task), "PredictionClassif")
})

test_that("trafo_fn and augment_fn are in the dictionary", {
  tbl = as.data.table(mlr_pipeops)
  expect_subset(c("trafo_fn", "augment_fn"), tbl$key)
})
