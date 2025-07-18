test_that("works with one ingress", {
  nn_one_layer = nn_module("nn_one_layer",
    initialize = function(task, size_hidden) {
      self$first = nn_linear(task$n_features, size_hidden)
      self$second = nn_linear(size_hidden, output_dim_for(task))
    },
    forward = function(x) {
      x = self$first(x)
      x = nnf_relu(x)
      self$second(x)
    }
  )
  learner = lrn("classif.module",
    module_generator = nn_one_layer,
    ingress_tokens = list(x = ingress_num()),
    epochs = 10,
    size_hidden = 20,
    batch_size = 16
  )
  task = tsk("iris")
  expect_learner_torch(learner, task = task)
  expect_true("size_hidden" %in% learner$param_set$ids())
  learner$train(task)
  learner$network
  expect_class(learner$model, "learner_torch_model")
})

test_that("works with multiple ingress", {
  nn_two_inputs = nn_module("nn_two_inputs",
    initialize = function(task, size_hidden) {
      types = task$feature_types$type
      self$first_num = nn_linear(length(types[types %in% c("numeric", "integer")]), size_hidden)
      self$first_categ = nn_linear(length(types[types %in% c("factor", "ordered", "logical")]), size_hidden)
      self$head = nn_linear(size_hidden * 2, output_dim_for(task))
    },
    forward = function(x_num, x_categ) {
      x_num = self$first_num(x_num)
      x_categ = self$first_categ(x_categ$to(dtype = torch_float32()))
      x = torch_cat(list(x_num, x_categ), dim = 2)
      x = nnf_relu(x)
      self$head(x)
    }
  )

  learner = lrn("classif.module",
    module_generator = nn_two_inputs,
    ingress_tokens = list(x_num = ingress_num(), x_categ = ingress_categ()),
    epochs = 10,
    size_hidden = 20,
    batch_size = 16
  )
  task = tsk("german_credit")
  expect_learner_torch(learner, task = task)
  learner$train(task)
  learner$network
  expect_class(learner$model, "learner_torch_model")
})

test_that("works with function", {
  learner = lrn("regr.module", module_generator = function(task) {
    nn_linear(task$n_features, 1)
  }, epochs = 1, batch_size = 50, ingress_tokens = list(x = ingress_num()))
  task = tsk("mtcars")
  learner$train(task)
  expect_class(learner$network, "nn_linear")
  expect_prediction(learner$predict(task))
})

