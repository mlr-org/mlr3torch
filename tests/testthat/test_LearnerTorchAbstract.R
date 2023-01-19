LearnerClassifTest = R6Class("LearnerClassifTest",
  inherit = LearnerClassifTorchAbstract,
  public = list(
    initialize = function(optimizer = t_opt("adam"), loss = t_loss("cross_entropy"), param_vals = list()) {
      super$initialize(
        id = "classif.test",
        label = "Test Classifier",
        feature_types = c("numeric", "integer"),
        param_set = ps(),
        properties = c("weights", "hotstart_forward", "multiclass", "twoclass"),
        optimizer = optimizer,
        loss = loss,
        man = "mlr3torch::mlr_learners_classif.test"
      )
    }
  ),
  private = list(
    .network = function(task) {
      nn_linear(length(task$feature_names), length(task$target_names))
    },
    .dataloader = function(param_vals, task) {}
  )
)


test_that("LearnerTorchAbstract manual", {
  l = LearnerClassifTest$new()
  l$param_set$values$epochs = 10L
  expect_r6(l, "Learner")
  expect_set_equal(l$properties, c("weights", "hotstart_forward", "multiclass", "twoclass"))

  task = tsk("iris")
  l$train(task)


  run_autotest(l)
})

test_that("LearnerTorchAbstract autotest", {
  l = LearnerClassifTest$new()
  expect_r6(l$param_set, l$param_set)
  expect_identical(l$param_)

  res = run_autotest(l)
})


# test that cloning works!



test_that("assemble and dissamble work.", {
  task = tsk("iris")
  f = function() {
    devtools::load_all("~/mlr/mlr3torch")
    learner = mlr3::lrn("classif.mlp",
      layers = 2L,
      p = 0.2,
      batch_size = 16L,
      epochs = 1L,
      d_hidden = 10,
      activation = "relu",
      optimizer = "adam"
    )
    task = mlr3::tsk("iris")

    learner$train(task)
    learner$serialize()
    saveRDS(learner, "~/.lol/loeerna.R")

  }
  callr::r(f)
  learner = readRDS("~/.lol/loeerna.R")
  learner$unserialize()
  learner$train(task)

  dir = tempfile(fileext = ".rds")
  saveRDS(learner, dir)
  learner_ = readRDS(dir)

  saveRDS(learner$s)
})
