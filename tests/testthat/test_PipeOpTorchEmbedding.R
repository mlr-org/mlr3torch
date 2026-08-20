categ_task = function() {
  po("select", selector = selector_type(c("factor", "ordered")))$train(list(tsk("german_credit")))[[1L]]
}

test_that("PipeOpTorchEmbedding autotest", {
  po_test = po("nn_embedding", embedding_dim = 5)
  task = categ_task()
  graph = po("torch_ingress_categ") %>>% po_test

  expect_pipeop_torch(graph, "nn_embedding", task)
})

test_that("PipeOpTorchEmbedding paramtest", {
  # `.weight` sets the table's initial values, which a `PipeOp` has no way to receive
  res = expect_paramset(po("nn_embedding"), nn_embedding, exclude = ".weight")
  expect_paramtest(res)
})

test_that("PipeOpTorchEmbeddingBag autotest", {
  po_test = po("nn_embedding_bag", embedding_dim = 5)
  task = categ_task()
  graph = po("torch_ingress_categ") %>>% po_test

  expect_pipeop_torch(graph, "nn_embedding_bag", task)
})

test_that("PipeOpTorchEmbeddingBag paramtest", {
  # `include_last_offset` only means something together with the `offsets` argument of the forward
  # method, which the tensors of a network do not carry
  res = expect_paramset(po("nn_embedding_bag"), nn_embedding_bag,
    exclude = c(".weight", "include_last_offset"))
  expect_paramtest(res)
})

test_that("the embedding adds a dimension and the bag reduces one", {
  embedding = po("nn_embedding", embedding_dim = 5, num_embeddings = 10)
  expect_equal(embedding$shapes_out(list(c(NA, 4L)))[[1L]], c(NA, 4L, 5L))
  # the input may be a sequence of token ids of any rank
  expect_equal(embedding$shapes_out(list(c(NA, 7L, 3L)))[[1L]], c(NA, 7L, 3L, 5L))
  # an unknown number of features stays unknown, the table's size does not depend on it
  expect_equal(embedding$shapes_out(list(c(NA, NA)))[[1L]], c(NA, NA, 5L))

  bag = po("nn_embedding_bag", embedding_dim = 5, num_embeddings = 10)
  expect_equal(bag$shapes_out(list(c(NA, 4L)))[[1L]], c(NA, 5L))
  expect_error(bag$shapes_out(list(c(NA, 4L, 3L))), "requires an input with 2 dimensions",
    fixed = TRUE)
})

test_that("'num_embeddings' is inferred from the task and can be set explicitly", {
  task = categ_task()
  cardinalities = categ_cardinalities(task)
  md = (po("torch_ingress_categ") %>>% po("nn_embedding", embedding_dim = 5))$train(task)[[1L]]
  expect_equal(md$graph$pipeops$nn_embedding$module$num_embeddings, max(cardinalities))

  # a value the user set wins over the task
  md2 = (po("torch_ingress_categ") %>>%
    po("nn_embedding", embedding_dim = 5, num_embeddings = 100))$train(task)[[1L]]
  expect_equal(md2$graph$pipeops$nn_embedding$module$num_embeddings, 100)
})

test_that("'num_embeddings' has to come from somewhere", {
  # without a task and without the parameter the table cannot be sized at all
  expect_error(po("nn_embedding", embedding_dim = 5)$shapes_out(list(c(NA, 4L))),
    "'num_embeddings' is not set and the task has no categorical features", fixed = TRUE)
  expect_error(
    po("nn_embedding", embedding_dim = 5)$shapes_out(list(c(NA, 4L)), task = tsk("iris")),
    "'num_embeddings' is not set and the task has no categorical features", fixed = TRUE)
  # a lazy tensor of token ids carries no cardinalities, so the parameter is required for it
  expect_equal(
    po("nn_embedding", embedding_dim = 5, num_embeddings = 30)$shapes_out(list(c(NA, 4L)))[[1L]],
    c(NA, 4L, 5L))
})

test_that("the embedding bag rejects the combinations that torch cannot back-propagate", {
  task = categ_task()
  graph = po("torch_ingress_categ") %>>%
    po("nn_embedding_bag", embedding_dim = 5, mode = "max", sparse = TRUE)
  expect_error(graph$train(task), "cannot be combined", fixed = TRUE)
})
