devtools::load_all()

# source(here::here("attic", "refactored-ft-transformer", "R", "nn_ft_transformer.R"))

library(tidytable)

# construct a task matching the pre-implementing tests
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

y = rbinom(n = 4, size = 1, prob = 0.5)

dt = bind_cols(y, dt_num, dt_cat) |>
  rename(y = ...1)
task = as_task_classif(dt, target = "y")

# sketch what the FT-Transformer as a graph should look like
path_num = po("select", id = "select_num", selector = selector_type("numeric")) %>>% po("torch_ingress_num") %>>% po("nn_tokenizer_num") 
path_categ = po("select", id = "select_categ", selector = selector_type("factor")) %>>% po("torch_ingress_categ") %>>% po("nn_tokenizer_categ")

# This doesn't work because the tokenizers output a ModelDescriptor
graph_tokenizer = gunion(list(path_num, path_categ)) %>>%
  po("nn_merge_cat")

# TODO: access x[, -1] first
# TODO: sometimes there is no normalization, i.e. nn_identity instead of nn_layer_norm, figure out how to handle this
graph_head = po("nn_layer_norm") %>>%
  po("nn_relu") %>>%
  po("nn_linear") %>>%
  po("nn_head")

# taking into account the different handlings of the first and last layers
# TODO: determine where all those configurations go (i.e. the stuff defined in make_baseline, etc.)
graph_ft_transformer = graph_tokenizer %>>%
  po("cls") %>>%
  po("transformer_layer", first_layer = TRUE, first_prenormalization = FALSE) %>>%
  po("nn_block", po_transformer, n_blocks = 3) %>>%
  po("transformer_layer", last_layer_query_idx = 1L) %>>%
  graph_head
