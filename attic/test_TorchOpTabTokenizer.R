# TODO: Check with concrete numbers that everything works as intented
test_that("nn_tokenizer_numeric works", {
  n = 16
  d_token = 3
  bias = TRUE
  n_features = 2
  tokenizer = nn_tokenizer_numeric(n_features, d_token, bias)
  input = torch_randn(n, n_features)
  tokens = tokenizer(input)
  expect_true(all(dim(tokens) == c(n, n_features, d_token)))
})


test_that("nn_tokenizer_categorical works", {
  cardinalities = c(10, 2, 4)
  d_token = 3
  bias = TRUE
  n = 16
  tokenizer = nn_tokenizer_categorical(cardinalities, d_token, bias)
  col1 = torch_randint(low = 1L, high = cardinalities[1L] + 1L, size = n)[.., NULL]
  col2 = torch_randint(low = 1L, high = cardinalities[2L] + 1L, size = n)[.., NULL]
  col3 = torch_randint(low = 1L, high = cardinalities[3L] + 1L, size = n)[.., NULL]
  input = torch_cat(list(col1, col2, col3), 2L)$to(torch_long())
  tokens = tokenizer(input)
  expect_true(all(dim(tokens) == c(n, length(cardinalities), d_token)))

})

# TODO: add edge cases: no numeric and no categorical features
test_that("nn_tokenizer works", {
  n = 16
  d_token = 3
  bias = TRUE
  n_features = 2
  cardinalities = c(10, 2, 4)
  cls = FALSE
  cols = list()
  for (i in seq_along(cardinalities)) {
    col = torch_randint(low = 1L, high = cardinalities[i] + 1L, size = n)[.., NULL]
    cols[[i]] = col
  }

  input_cat = torch_cat(cols, 2L)$to(torch_long())
  input_num = torch_randn(n, n_features)

  tokenizer = nn_tab_tokenizer(n_features, cardinalities, d_token, bias, cls)
  output = tokenizer(list(num = input_num, cat = input_categ))
  expect_true(all(dim(output) == c(n, n_features + length(cardinalities), d_token)))
})

test_that("TorchOpTabTokenizer works", {
  d_token = 7
  task = tsk("boston_housing")
  graph = top("input") %>>%
    top("tab_tokenizer", d_token = d_token)
  expect_error(graph, regexp = NA)
})
