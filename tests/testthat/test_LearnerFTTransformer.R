make_ft_transformer = function(task_type, ...) {
  params = list(
     attention_n_heads = 1,
     attention_dropout = 0.1,
     ffn_d_hidden = 100,
     ffn_dropout = 0.1,
     ffn_activation = nn_reglu,
     residual_dropout = 0.0,
     prenormalization = TRUE,
     is_first_layer = TRUE,
     attention_initialization = "kaiming",
     ffn_normalization = nn_layer_norm,
     attention_normalization = nn_layer_norm,
     query_idx = NULL,
     attention_bias = TRUE,
     ffn_bias_first = TRUE,
     ffn_bias_second = TRUE,

     epochs = 1L,
     batch_size = 32L,
     n_blocks = 1L,
     d_token = 10L
  )
  params = insert_named(params, list(...))
  invoke(lrn, .key = sprintf("%s.ft_transformer", task_type), .args = params)
}

no_wd = function(name) {
  # implementation from paper description
  linear_bias_param = grepl("linear_", name, fixed = TRUE) && grepl(".bias", name, fixed = TRUE)

  other_no_wd_params = c("embedding", "_normalization")

  return(
    any(map_lgl(other_no_wd_params, function(pattern) grepl(pattern, name, fixed = TRUE)))
    || linear_bias_param
  )

  # implementation in https://github.com/yandex-research/rtdl-revisiting-models/blob/main/package/rtdl_revisiting_models.py
  # no_wd_params = c("embedding", "_normalization", ".bias")

  # return(any(map_lgl(no_wd_params, function(pattern) grepl(pattern, name, fixed = TRUE))))
}

rtdl_param_groups = function(parameters) {
  no_wd_idx = map_lgl(names(parameters), no_wd)
  no_wd_group = parameters[no_wd_idx]

  main_group = parameters[!no_wd_idx]

  list(
    list(params = main_group),
    list(params = no_wd_group, weight_decay = 0)
  )
}

test_that("param groups work", {
  learner = make_ft_transformer("classif")
  default_weight_decay = 0.23
  learner$param_set$set_values(opt.weight_decay = default_weight_decay)
  learner$param_set$set_values(opt.param_groups = rtdl_param_groups)

  task = tsk("german_credit")$filter(1:10)
  learner$train(task)

  expect_equal(length(learner$model$optimizer$param_groups), 2L)
  expect_equal(learner$model$optimizer$param_groups[[1L]]$weight_decay, default_weight_decay)
  expect_equal(learner$model$optimizer$param_groups[[2L]]$weight_decay, 0)

  expect_learner(learner)
})

test_that("basic functionality", {
  learner = make_ft_transformer("classif")
  task = tsk("german_credit")$filter(1:10)
  learner$train(task)

  expect_learner(learner)
})

test_that("works with only numeric input", {
  learner = make_ft_transformer("classif")
  task = tsk("iris")
  learner$train(task)

  expect_learner(learner)
})

test_that("works with only categorical input", {
  learner = make_ft_transformer("classif")
  task = tsk("german_credit")$filter(1:10)
  task$select(c("credit_history", "employment_duration", "foreign_worker"))

  learner$train(task)

  expect_learner(learner)
})

test_that("works with lazy tensors", {
  task = as_task_regr(data.table(
      x_categ = as_lazy_tensor(matrix(rep(1:10, 20), ncol = 1)),
      x_num = as_lazy_tensor(matrix(runif(200), ncol = 2)),
      y = rnorm(100)
    ), target = "y", id = "test")

  learner = make_ft_transformer("regr",
    ingress_tokens = list(
      num.input = ingress_ltnsr("x_num"),
      categ.input = ingress_ltnsr("x_categ")
    ),
    cardinalities = 20
  )
  learner$train(task)

  expect_learner(learner)
})

make_ft_transformer_default = function(task_type, ...) {
  params = list(
     epochs = 1L,
     batch_size = 32L,
     n_blocks = 4L,
     d_token = 192L,
     ffn_d_hidden_multiplier = 4 / 3
  )
  params = insert_named(params, list(...))
  invoke(lrn, .key = sprintf("%s.ft_transformer", task_type), .args = params)
}

test_that("defaults work", {
  lrn = make_ft_transformer_default("classif")
  task = tsk("iris")$filter(1:20)
  lrn$train(task)

  expect_learner(lrn)
})
