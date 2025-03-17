# a bunch of PipeOpModules for everything
# https://mlr3torch.mlr-org.com/reference/mlr_pipeops_module.html

# but we also need to write tests for the new... thing.... correct?

library(torch)
library(mlr3)
library(data.table)
library(tidytable)
library(mlr3pipelines)
library(mlr3torch)

source(here::here("attic", "refactored-ft-transformer", "nn_ft_transformer.R"))

# as first pass, simply use PipeOpModule to wrap the existing nn_modules
po_ft_transformer = po("module",
  id = "ft-transformer",
  module = make_baseline(
    n_num_features=3,
    cat_cardinalities=c(2, 3),
    d_token=8,
    n_blocks=2,
    attention_dropout=0.2,
    ffn_d_hidden=6,
    ffn_dropout=0.2,
    residual_dropout=0.0,
    d_out=1
  )
)

po_transformer_block = po("module",

)

po_cls_token = po("module", 
)

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
path_num = po("select", selector = selector_type("numeric")) %>>% po("torch_ingress_num") %>>% po("nn_tokenizer_num") 
path_categ = po("select", selector = selector_type("factor")) %>>% po("torch_ingress_categ") %>>% po("nn_tokenizer_categ")

graph_tokenizer = gunion(list(path_num, path_categ)) %>>%
  po("featureunion")

po_transformer = po("transformer_layer")

# TODO: access x[, -1] first
graph_head = po("trafo_normalize") %>>%
  po("nn_relu") %>>%
  po("nn_linear")

# original proposal 
graph_tokenizer %>>%
  po("cls") %>>%
  po("nn_block", po_transformer, n_blocks = 5) %>>%
  graph_head

# taking into account the different handlings of the first and last layers
graph_tokenizer %>>%
  po("cls") %>>%
  po("transformer_layer", prenorm = FALSE) %>>%
  po("nn_block", po_transformer, n_blocks = 3) %>>%
  po("transformer_layer", last_layer = TRUE) %>>%
  graph_head

# alternative requiring the implementation of a new PipeOpTransformerBlock  
graph_tokenizer %>>%
  po("cls") %>>%
  po("transformer_block", n_layers = 5) %>>%
  graph_head
