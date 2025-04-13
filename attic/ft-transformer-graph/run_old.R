source("../old-ft-transformer/R/nn_ft_transformer.R")

nn_ft_transformer_module = make_baseline(
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

n_objects = 4
n_num_features = 3
n_cat_features = 2
d_token = 7
x_num = torch_randn(n_objects, n_num_features)

mat = matrix(nrow=4, ncol=2)
mat[1, ] = c(1L, 2L)
mat[2, ] = c(2L, 1L)
mat[3, ] = c(1L, 3L)
mat[4, ] = c(2L, 2L)
x_cat = torch_tensor(mat)

old_output = nn_ft_transformer_module(x_num, x_cat)

print(old_output)
print(old_output$shape)
