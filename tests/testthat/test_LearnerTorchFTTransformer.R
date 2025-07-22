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

# make sure this matches mlr3tuningspaces
no_wd = function(name) {
  # TODO: refactor, since we call it "Tokenizer", so the module does not have "embedding" in the name
  # furthermore, the tokenizer modules seem to end up unnamed anyway
  # this will also disable weight decay for the input projection bias of the attention heads
  # ()
  no_wd_params = c("_normalization", "bias")

  return(any(map_lgl(no_wd_params, function(pattern) grepl(pattern, name, fixed = TRUE))))
}

rtdl_param_groups = function(parameters) {
  ffn_norm_idx = grepl("ffn_normalization", names(parameters), fixed = TRUE)
  ffn_norm_num_in_module_list = as.integer(strsplit(names(parameters)[ffn_norm_idx][1], ".", fixed = TRUE)[[1]][2])
  cls_num_in_module_list = ffn_norm_num_in_module_list - 1
  nums_in_module_list = sapply(strsplit(names(parameters), ".", fixed = TRUE), function(x) as.integer(x[2]))
  tokenizer_idx = nums_in_module_list < cls_num_in_module_list

  no_wd_idx = map_lgl(names(parameters), no_wd) | tokenizer_idx
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
  
  # german credit: both categorical and numeric
  task = tsk("german_credit")$filter(1:10)
  learner$train(task)

  # mtcars: only numeric
  learner_regr = make_ft_transformer("regr")
  learner_regr$param_set$set_values(opt.weight_decay = default_weight_decay)
  learner_regr$param_set$set_values(opt.param_groups = rtdl_param_groups)
  task_num = tsk("mtcars")
  learner_regr$train(task_num)

  task_categ = tsk("penguins")$select(c("island", "sex"))
  complete_cases_idx = which(complete.cases(task_categ$data()))
  task_categ$filter(complete_cases_idx)
  learner$train(task_categ)

  # TODO: add an assertion on the indices for the params
  # it should be clear which params end up in which param groups
  # and this will test that "everything" ends up in the right place
  # or at the very least, 
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
