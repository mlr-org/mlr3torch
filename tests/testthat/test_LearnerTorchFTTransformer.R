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
     n_blocks = 3L,
     d_token = 10L
  )
  params = insert_named(params, list(...))
  invoke(lrn, .key = sprintf("%s.ft_transformer", task_type), .args = params)
}

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

test_that("marshaling works (#412)", {
  lrn = make_ft_transformer("classif")
  task = tsk("german_credit")$select("credit_history")
  lrn$train(task)
  lrn$marshal()$unmarshal()
  gc()
  expect_class(lrn$network$buffers[[1L]], "torch_tensor")
})