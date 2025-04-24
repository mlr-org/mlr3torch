make_ft_transformer = function(task_type, ...) {
  params = list(
     attention_n_heads = 1,
     attention_dropout = 0.1,
     ffn_d_hidden = 100,
     ffn_dropout = 0.1,
     ffn_activation = nn_reglu(),
     residual_dropout = 0.0,
     prenormalization = TRUE,
     first_prenormalization = FALSE,
     is_first_layer = TRUE,
     attention_initialization = "kaiming",
     ffn_normalization = nn_layer_norm,
     attention_normalization = nn_layer_norm,
     query_idx = NULL,
     kv_compression_ratio = 1.0,
     kv_compression_sharing = "headwise",

     # training
     epochs = 1L,
     batch_size = 32L,
     n_blocks = 1L,
     d_token = 10L
  )
  params = insert_named(params, list(...))
  invoke(lrn, .key = sprintf("%s.ft_transformer", task_type), .args = params)
}
test_that("basic functionality", {
  learner = make_ft_transformer("classif")
  task = tsk("german_credit")$filter(1:10)
  learner$train(task)
})

test_that("jitting works and is the same as non-jitted", {
  # TODO: For some reason this fails.
  # When I remove the ft_cls token
  # This is because the $expand() method of CLS does some reshaping dynamically.
  # Maybe we can rewrite this

  # TOOD: Also check that the results of jitting and non jitting are the same
  # IF not, disable the jit_trace parameter

  learner = make_ft_transformer("classif", jit_trace = TRUE, n_blocks = 1, drop_last = FALSE)
  task = tsk("german_credit")$filter(1:10)
  learner$train(task)

})

test_that("works with only categorical/only numeric", {
  # TODO:
})

test_that("works with lazy tensors", {
  task = as_task_regr(data.table(
      x_categ = as_lazy_tensor(matrix(1:100, ncol = 1)),
      x_num = as_lazy_tensor(matrix(runif(200), ncol = 2)),
      y = rnorm(100)
    ), target = "y", id = "test")

  learner = make_ft_transformer("regr", input_map = c(num.input = "x_num", categ.input = "x_categ"))
  learner$train(task)
})
