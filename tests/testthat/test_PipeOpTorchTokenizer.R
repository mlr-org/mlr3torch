test_that("nn_tokenizer_num works properly", {
  x = torch_randn(4, 2)
  n_objects = x$shape[1]
  n_features = x$shape[2]
  d_token = 3
  tokenizer = nn_tokenizer_num(n_features, d_token, TRUE, "uniform")
  tokens = tokenizer(x)
  expect_true(all(tokens$shape == c(n_objects, n_features, d_token)))
})

test_that("pipeop numeric tokenizer", {
  po_tokenize = po("nn_tokenizer_num", d_token = 10)
  graph = po("torch_ingress_num") %>>% po_tokenize
  task = tsk("iris")
  expect_pipeop_torch(graph, "nn_tokenizer_num", task, "nn_tokenizer_num")
  res = expect_paramset(po_tokenize, nn_tokenizer_num, exclude = "n_features")
  expect_paramtest(res)
})

test_that("nn_tokenizer_categ works properly", {
  cardinalities = c(3, 10)
  mat = matrix(nrow = 4, ncol = 2)
  mat[1, ] = c(1L, 6L)
  mat[2, ] = c(2L, 8L)
  mat[3, ] = c(1L, 3L)
  mat[4, ] = c(3L, 5L)
  x = torch_tensor(mat)
  n_objects = x$shape[1]
  n_features = x$shape[2]
  d_token = 3
  tokenizer = nn_tokenizer_categ(cardinalities, d_token, TRUE, "uniform")
  tokens = tokenizer(x)
  expect_true(all(tokens$shape == c(n_objects, n_features, d_token)))
})

test_that("pipeop categ tokenizer", {
  po_tokenize = po("nn_tokenizer_categ", d_token = 10)
  graph = po("torch_ingress_categ") %>>% po_tokenize
  task = tsk("breast_cancer")
  expect_pipeop_torch(graph, "nn_tokenizer_categ", task, "nn_tokenizer_categ")
  res = expect_paramset(po_tokenize, nn_tokenizer_categ, exclude = "cardinalities")
  expect_paramtest(res)
})
