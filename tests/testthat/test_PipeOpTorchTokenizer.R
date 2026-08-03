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

test_that("nn_tokenizer_categ works with logical features", {
  # Regression test: logical features used to be encoded as 0/1 by batchgetter_categ() and
  # their cardinality was inferred as 0 (Task$levels() returns NULL for them). The first
  # meant indexing row 0 of a 1-based nn_embedding, the second an assertion failure.
  d = data.frame(
    y = factor(rep(c("a", "b"), 10)),
    lg = rep(c(TRUE, FALSE), 10),
    f = factor(rep(c("x", "y"), 10))
  )
  task = mlr3::as_task_classif(d, target = "y")

  graph = po("torch_ingress_categ") %>>% po("nn_tokenizer_categ", d_token = 4)
  md = graph$train(task)[[1L]]
  net = md$graph
  batch = batchgetter_categ(task$data(rows = 1:3, cols = c("lg", "f")))
  out = net$train(batch)[[1L]]
  expect_equal(out$shape, c(3, 2, 4))
  expect_true(all(is.finite(as.array(out))))

  # a task consisting only of logical features also works
  d2 = data.frame(y = factor(rep(c("a", "b"), 10)), lg = rep(c(TRUE, FALSE), 10))
  task2 = mlr3::as_task_classif(d2, target = "y")
  md2 = (po("torch_ingress_categ") %>>% po("nn_tokenizer_categ", d_token = 4))$train(task2)[[1L]]
  out2 = md2$graph$train(batchgetter_categ(task2$data(rows = 1:3, cols = "lg")))[[1L]]
  expect_equal(out2$shape, c(3, 1, 4))
})

test_that("shape inference needs the feature dimension", {
  expect_error(po("nn_tokenizer_num", d_token = 10)$shapes_out(list(c(NA, NA)), task = tsk("iris")),
    "requires the feature dimension (dimension 2)", fixed = TRUE)
})

test_that("the categorical tokenizer takes the number of tokens from the task", {
  # it does not read the input dimension: the number of tokens is the number of categorical
  # features, which comes from the task (or from `cardinalities`)
  expect_equal(po("nn_tokenizer_categ", d_token = 10)$shapes_out(list(c(NA, NA)),
    task = tsk("breast_cancer"))[[1L]], c(NA, 9L, 10L))
  expect_equal(po("nn_tokenizer_categ", d_token = 5L)$shapes_out(list(c(3L, 1L)),
    task = tsk("breast_cancer"))[[1L]], c(3L, 9L, 5L))

  # the module agrees (it needs an integer tensor, so it is checked directly)
  obj = po("nn_tokenizer_categ", d_token = 5L)
  module = get_private(obj)$.make_module(list(c(3L, 1L)), obj$param_set$get_values(), tsk("breast_cancer"))
  expect_equal(dim(with_no_grad(module(torch_ones(3L, 1L, dtype = torch_long())))), c(3, 9, 5))
})

test_that("the number of tokens is never unknown", {
  # without a task and without 'cardinalities' the module cannot be built at all, so this is a
  # configuration error rather than a shape that is merely not known yet
  obj = po("nn_tokenizer_categ", d_token = 5L)
  expect_error(obj$shapes_out(list(c(NA, 9L))), "neither a task nor the 'cardinalities' parameter",
    fixed = TRUE)
  # ... and the module says the same when it is built directly
  expect_error(get_private(obj)$.shape_dependent_params(list(c(NA, 9L)), obj$param_set$get_values(), NULL),
    "neither a task nor the 'cardinalities' parameter", fixed = TRUE)
  # 'cardinalities' alone is enough
  expect_equal(po("nn_tokenizer_categ", d_token = 5L, cardinalities = c(2L, 3L))$
    shapes_out(list(c(NA, 2L)))[[1L]], c(NA, 2L, 5L))
})

test_that("shape inference agrees with the module for random shapes and parameters", {
  # the categorical tokenizer needs an integer tensor and a task, so it is checked above instead
  expect_shape_inference_sampled("nn_tokenizer_num",
    list(rank = 2L, params = function() list(d_token = sample(2:6, 1L))))
})
