library(torch)

# test for numeric tokenizer
x = torch_randn(4, 2)
n_objects = x$shape[1]
n_features = x$shape[2]
d_token = 3
tokenizer = nn_tokenizer_numeric(n_features, d_token, TRUE)
tokens = tokenizer(x)
assert_true(all(tokens$shape == c(n_objects, n_features, d_token)))


# test for categorical tokenizer
cardinalities = c(3, 10)
mat = matrix(nrow=4, ncol=2)
mat[1, ] = c(1L, 6L)
mat[2, ] = c(2L, 8L)
mat[3, ] = c(1L, 3L)
mat[4, ] = c(3L, 5L)
x = torch_tensor(mat)
n_objects = x$shape[1]
n_features = x$shape[2]
d_token = 3
tokenizer = nn_tokenizer_categorical(cardinalities, d_token, TRUE)
tokens = tokenizer(x)
assert_true(all(tokens$shape == c(n_objects, n_features, d_token)))


# test for tab tokenizer
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
tokenizer = nn_tab_tokenizer(n_num_features, c(2, 3), d_token, TRUE, FALSE)
tokens = tokenizer(x_num, x_cat)
assert_true(all(tokens$shape == c(n_objects, n_num_features + n_cat_features, d_token)))


# test for clstoken
batch_size = 2
n_tokens = 3
d_token = 4
cls_token = nn_cls_token(d_token)
x = torch_randn(batch_size, n_tokens, d_token)
x = cls_token(x)
assert_true(all(x$shape == c(batch_size, n_tokens + 1, d_token)))
assert_true(all(x[, -1, ] == cls_token$expand(x$shape[1]))) # does not work but they are the same



# test for reglu
module = nn_reglu()
x = torch_randn(3, 4)
assert_true(all(module(x)$shape == c(3, 2)))


# test for geglu
module = nn_geglu()
x = torch_randn(3, 4)
assert_true(all(module(x)$shape == c(3, 2)))


### test for multihead attention
n_objects = 2
n_tokens =  3
d_token = 12
n_heads = 6
a = torch_randn(n_objects, n_tokens, d_token)
b = torch_randn(n_objects, n_tokens * 2, d_token)
module = nn_multi_head_attention(
  d_token=d_token, n_heads=n_heads, dropout=0.2, bias=TRUE, initialization='kaiming'
)
# self-attention
x_attention_stats = module(a, a, NULL, NULL)
assert_true(all(x_attention_stats[['x']]$shape == a$shape))
assert_true(all(x_attention_stats[['attention_probs']]$shape == c(n_objects * n_heads, n_tokens, n_tokens)))
assert_true(all(x_attention_stats[['attention_logits']]$shape == c(n_objects * n_heads, n_tokens, n_tokens)))
# cross-attention
module(a, b, NULL, NULL)
# Linformer self-attention with the 'headwise' sharing policy
k_compression = nn_linear(n_tokens, n_tokens %/% 4)
v_compression = nn_linear(n_tokens, n_tokens %/% 4)
module(a, a, k_compression, v_compression)
# Linformer self-attention with the 'key-value' sharing policy
kv_compression = nn_linear(n_tokens, n_tokens %/% 4)
module(a, a, kv_compression, kv_compression)




