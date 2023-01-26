test_that("autotest: classification", {
  module = nn_module(
    initialize = function(task) {
      out = if (task$task_type == "classif") length(task$class_names) else 1
      self$linear = nn_linear(length(task$feature_names), out)
    },
    forward = function(x) {
      self$linear(x)
    }
  )

  learner = lrn("classif.torch", module, feature_types = c("numeric", "integer"), 
    batch_size = 16, epochs = 5
  )
  # task = tsk("iris")
  # learner$train(task)
  # learner$predict_type = "prob"
  # learner$predict(task)
  #
  expect_learner(learner)

  result = run_autotest(learner, check_replicable = FALSE, exclude = "sanity")
  expect_true(result, info = result$error)
})

