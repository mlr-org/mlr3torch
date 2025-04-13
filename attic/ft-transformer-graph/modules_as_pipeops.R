devtools::load_all()

library(here)

source(here("attic", "ft-transformer-graph", "create_task.R"))

# prototype what you want the FT-Transformer to look like as a graph
# this should guide the implementations of the individual PipeOps

# as a first pass, fix some sensible default parameters
# i.e.

# TODO: access x[, -1] first. Implement a PipeOp for this.
# TODO: sometimes there is no normalization, i.e. nn_identity instead of nn_layer_norm, figure out how to handle this
# analogous: nn_ft_head
graph_head = po("nn_fn", fn = function(x) x[, -1]) %>>%
  po("nn_layer_norm", dims = 1) %>>%
  po("nn_relu") %>>%
  po("nn_linear", id = "linear_head", out_features = 1) %>>%
  po("nn_head")

# begin Copilot
d_token = 32
attention_n_heads = 8
ffn_d_hidden = 64

# Update paths with proper parameters

# with selector
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
  po("nn_merge_cat", param_vals = list(dim = 2))  # merge along sequence dimension

# without selector
# path_num = po("torch_ingress_num") %>>%
#   po("nn_tokenizer_num", param_vals = list(
#     d_token = d_token,
#     bias = TRUE,
#     initialization = "uniform"
#   ))

# path_categ = po("torch_ingress_categ") %>>%
#   po("nn_tokenizer_categ", param_vals = list(
#     d_token = d_token,
#     bias = TRUE,
#     initialization = "uniform"
#   ))

# graph_tokenizer = gunion(list(path_num, path_categ)) %>>%
#   po("nn_merge_cat", param_vals = list(dim = 2))  # merge along sequence dimension

# Update transformer configuration
po_transformer = po("transformer_layer",
  id = "intermediate_transformer_layer",
  param_vals = list(
    d_token = d_token,
    attention_n_heads = attention_n_heads,
    attention_dropout = 0.1,
    ffn_activation = nn_reglu(), # TODO: factor out
    ffn_d_hidden = ffn_d_hidden,
    ffn_dropout = 0.1,
    residual_dropout = 0.0,
    prenormalization = TRUE,
    attention_initialization = "kaiming",
    ffn_normalization = nn_layer_norm,
    attention_normalization = nn_layer_norm,
    query_idx = NULL
  )
)

# Update the full transformer graph
graph_ft_transformer = graph_tokenizer %>>%
  po("cls", d_token = d_token, initialization = "uniform") %>>%
  po("transformer_layer",
    id = "first_transformer_layer",
    param_vals = list(
      d_token = d_token,
      attention_n_heads = attention_n_heads,
      attention_dropout = 0.1,
      ffn_d_hidden = ffn_d_hidden,
      ffn_dropout = 0.1,
      ffn_activation = nn_reglu(),
      residual_dropout = 0.0,
      prenormalization = TRUE,
      is_first_layer = TRUE,
      first_prenormalization = FALSE,
      attention_initialization = "kaiming",
      ffn_normalization = nn_layer_norm,
      attention_normalization = nn_layer_norm,
      query_idx = NULL
    )
  ) %>>%
  po("nn_block", po_transformer, n_blocks = 3) %>>%
  po("transformer_layer",
    id = "last_transformer_layer",
    param_vals = list(
      d_token = d_token,
      attention_n_heads = attention_n_heads,
      attention_dropout = 0.1,
      ffn_d_hidden = ffn_d_hidden,
      ffn_dropout = 0.1,
      ffn_activation = nn_reglu(),
      residual_dropout = 0.0,
      prenormalization = TRUE,
      last_layer_query_idx = 1L,
      query_idx = 1L,
      attention_initialization = "kaiming",
      ffn_normalization = nn_layer_norm,
      attention_normalization = nn_layer_norm
    )
  ) %>>%
  graph_head

# lrn_ft_transformer = as_learner(graph_ft_transformer)

md_ft_transformer = graph_ft_transformer$train(task)[[1]]
# glrn_ft_transformer = as_learner(graph_from_trained_md)

# glrn_ft_transformer$train(task)
# graph_ft_transformer$predict(task, splits$test)

n_objects = 4
n_num_features = 3
n_cat_features = 2
d_token = 7
x_num = torch_randn(n_objects, n_num_features)

nn_ft_transformer_mlr3torch = nn_graph(md_ft_transformer$graph,
  shapes_in = list(torch_ingress_num.input = c(NA, n_num_features),
    torch_ingress_categ.input = c(NA, n_cat_features)
  )
)

mat = matrix(nrow=4, ncol=2)
mat[1, ] = c(1L, 2L)
mat[2, ] = c(2L, 1L)
mat[3, ] = c(1L, 3L)
mat[4, ] = c(2L, 2L)
x_cat = torch_tensor(mat)

# x = torch_cat(list(x_num, x_cat), dim = 2)

out = nn_ft_transformer_mlr3torch(x_num, x_cat)
out$shape
