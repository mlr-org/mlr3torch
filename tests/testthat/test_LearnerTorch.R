test_that("Correct error when trying to create deep clone of trained network", {
  learner = lrn("classif.torch_featureless")
  learner$param_set$set_values(epochs = 1, batch_size = 1)
  task = tsk("iris")
  learner$train(task)
  expect_error(learner$clone(deep = TRUE), regexp = "Deep clone of trained network is currently not supported")
})


test_that("Basic tests: Classification", {
  learner = LearnerTorchTest1$new(task_type = "classif")
  expect_class(learner, c("LearnerTorchTest1", "LearnerTorch", "Learner"))
  expect_equal(learner$id, "classif.test1")
  expect_equal(learner$label, "Test1 Learner")
  expect_set_equal(learner$feature_types, c("numeric", "integer"))
  expect_set_equal(learner$properties, c("multiclass", "twoclass"))

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
  expect_set_equal(learner$properties, c())

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

  # TODO: regex
  expect_error(helper(ps(loss.weight = p_dbl())))
  expect_error(helper(ps(opt.weight = p_dbl())))
  expect_error(helper(ps(cb.weight = p_dbl())))
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


# test_that("Bundling works",{
#   callr::r(function() {
#     library(mlr3torch)
#     module =
#     learner = lrn()
#   })
# })
#
# test_that("assemble and dissamble work.", {
#   callr::r(function() {
#     library(mlr3torch)
#     learner = lrn("")
#
#
#   })
#   f = function() {
#     library(mlr3torch)
#     devtools::load_all("~/mlr/mlr3torch")
#     learner = mlr3::lrn("classif.mlp",
#       layers = 2L,
#       p = 0.2,
#       batch_size = 16L,
#       epochs = 1L,
#       d_hidden = 10,
#       activation = "relu",
#       optimizer = "adam"
#     )
#     task = mlr3::tsk("iris")
#
#     learner$train(task)
#     learner$serialize()
#     saveRDS(learner, "~/.lol/loeerna.R")
#
#   }
#   callr::r(f)
#   learner = readRDS("~/.lol/loeerna.R")
#   learner$unserialize()
#   learner$train(task)
#
#   dir = tempfile(fileext = ".rds")
#   saveRDS(learner, dir)
#   learner_ = readRDS(dir)
#
#   saveRDS(learner$s)
# })

