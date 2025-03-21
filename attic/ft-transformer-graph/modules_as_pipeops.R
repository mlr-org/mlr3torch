devtools::load_all()

library(tidytable)

# construct a task matching the pre-implemented tests
x_num = torch_randn(4, 3)
dt_num = setNames(as.data.table(as_array(x_num)), c("Num1", "Num2", "Num3"))
mat = matrix(nrow=4, ncol=2)
mat[1, ] = c(1L, 2L)
mat[2, ] = c(2L, 1L)
mat[3, ] = c(1L, 3L)
mat[4, ] = c(2L, 2L)
x_cat = torch_tensor(mat)
dt_cat = as.data.table(as_array(x_cat)) |>
  mutate(across(everything(), as.factor)) |>
  setNames(c("Cat1", "Cat2"))

y = factor(rbinom(n = 4, size = 1, prob = 0.5), levels = c(0, 1))

dt = bind_cols(y, dt_num, dt_cat) |>
  rename(y = ...1)
task = as_task_classif(dt, target = "y")

d_embedding = 32

# TODO: access x[, -1] first. Implement a PipeOp for this.
# TODO: sometimes there is no normalization, i.e. nn_identity instead of nn_layer_norm, figure out how to handle this
graph_head = po("nn_layer_norm", dims = 1) %>>%
  po("nn_relu") %>>%
  po("nn_linear", id = "linear_head", out_features = 1) %>>%
  po("nn_head")

# begin Copilot
d_token = 32
attention_n_heads = 8
ffn_d_hidden = 64

# Update paths with proper parameters
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

# Create tokenizer graph with proper merge
graph_tokenizer = gunion(list(path_num, path_categ)) %>>%
  po("nn_merge_cat", param_vals = list(dim = 2))  # merge along sequence dimension

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
    attention_normalization = nn_layer_norm
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
      attention_normalization = nn_layer_norm
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
      attention_initialization = "kaiming",
      ffn_normalization = nn_layer_norm,
      attention_normalization = nn_layer_norm
    )
  ) %>>%
  graph_head
# end Copilot

graph_ft_transformer$train(task)

# TODO: fix missing word ("in") in the error message
# Error in .__PipeOpTorchHead__.shapes_out(self = self, private = private,  : 
#   Assertion on 'length(shapes_in[[1]]) == 2L' failed: Must be TRUE.
# This happened PipeOp nn_head's $train()
