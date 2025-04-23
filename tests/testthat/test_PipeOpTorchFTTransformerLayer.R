# test_that("PipeOpTorchFTTransformerLayer works on a simple example", {

# })

test_that("Entire FT-Transformer can be constructed as a graph", {
  # construct task
  torch_manual_seed(1)
  n_obs = 4
  n_num_features = 3
  n_cat_features = 2

  x_num = torch_randn(n_obs, n_num_features)
  dt_num = setNames(as.data.table(as_array(x_num)), c("Num1", "Num2", "Num3"))

  mat = matrix(nrow = n_obs, ncol = n_cat_features)
  mat[1, ] = c(1L, 2L)
  mat[2, ] = c(2L, 1L)
  mat[3, ] = c(1L, 3L)
  mat[4, ] = c(2L, 2L)
  x_cat = torch_tensor(mat)
  dt_cat = as.data.table(as_array(x_cat))
  dt_cat = dt_cat[, lapply(.SD, as.factor)]
  dt_cat = set_names(dt_cat, c("Cat1", "Cat2"))

  set.seed(1)
  y = factor(rbinom(n = 4, size = 1, prob = 0.5), levels = c(0, 1))
  dt = cbind(y, dt_num, dt_cat)
  task = as_task_classif(dt, target = "y")

  d_token = 32
  attention_n_heads = 8
  ffn_d_hidden = 64

  path_num = po("select", id = "select_num", selector = selector_type("numeric")) %>>%
    po("torch_ingress_num") %>>%
    po("nn_tokenizer_num", param_vals = list(
      d_token = d_token,
      bias = TRUE,
      initialization = "uniform"
    ))

  path_categ = po("select", id = "select_categ", selector = selector_type("factor")) %>>%
    po("torch_ingress_categ") %>>%
    po("nn_tokenizer_categ", param_vals = list(
      d_token = d_token,
      bias = TRUE,
      initialization = "uniform"
    ))

  graph_tokenizer = gunion(list(path_num, path_categ)) %>>%
    po("nn_merge_cat", param_vals = list(dim = 2))

  po_transformer = po("nn_ft_transformer_layer",
    id = "intermediate_transformer_layer",
    param_vals = list(
      attention_n_heads = attention_n_heads,
      attention_dropout = 0.1,
      ffn_activation = nn_reglu(),
      ffn_d_hidden = ffn_d_hidden,
      ffn_dropout = 0.1,
      residual_dropout = 0.0,
      prenormalization = TRUE,
      attention_initialization = "kaiming",
      ffn_normalization = nn_layer_norm,
      attention_normalization = nn_layer_norm,
      # test fails when this is not set, but the parameter has a default value...
      first_prenormalization = FALSE,
      is_first_layer = FALSE,
      query_idx = NULL,
      kv_compression_ratio = 1.0,
      kv_compression_sharing = "headwise"
    )
  )

  graph_output_head = po("nn_fn", fn = function(x) x[, -1]) %>>%
    po("nn_layer_norm", dims = 1) %>>%
    po("nn_relu") %>>%
    po("nn_linear", id = "linear_head", out_features = 1) %>>%
    po("nn_head")

  graph_ft_transformer = graph_tokenizer %>>%
    po("nn_ft_cls", initialization = "uniform") %>>%
    po("nn_ft_transformer_layer",
      id = "first_transformer_layer",
      param_vals = list(
        attention_n_heads = attention_n_heads,
        attention_dropout = 0.1,
        ffn_d_hidden = ffn_d_hidden,
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
        kv_compression_sharing = "headwise"
      )
    ) %>>%
    po("nn_block", po_transformer, n_blocks = 3) %>>%
    po("nn_ft_transformer_layer",
      id = "last_transformer_layer",
      param_vals = list(
        attention_n_heads = attention_n_heads,
        attention_dropout = 0.1,
        ffn_d_hidden = ffn_d_hidden,
        ffn_dropout = 0.1,
        ffn_activation = nn_reglu(),
        residual_dropout = 0.0,
        prenormalization = TRUE,
        is_first_layer = FALSE,
        first_prenormalization = FALSE,
        last_layer_query_idx = 1L,
        query_idx = 1L,
        attention_initialization = "kaiming",
        ffn_normalization = nn_layer_norm,
        attention_normalization = nn_layer_norm,
        kv_compression_ratio = 1.0,
        kv_compression_sharing = "headwise"
      )
    ) %>>%
    graph_output_head

  md_ft_transformer = graph_ft_transformer$train(task)[[1]]

  nn_ft_transformer_mlr3torch = nn_graph(md_ft_transformer$graph,
    shapes_in = list(torch_ingress_num.input = c(NA, n_num_features),
      torch_ingress_categ.input = c(NA, n_cat_features)
    )
  )

  out = nn_ft_transformer_mlr3torch(x_num, x_cat)

  expect_equal(out$shape, c(4, 1))
})
