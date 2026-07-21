small = function(...) {
  list(d_token = 8L, depth = 1L, n_heads = 2L, dim_head = 4L, dim_head_row = 4L,
    d_hidden_num_embed = 8L, d_hidden_head = 8L, ...)
}

test_that("nn_saint forward works for all attention types", {
  for (at in c("col", "row", "colrow")) {
    net = invoke(nn_saint, n_features_num = 3L, cardinalities = c(4L, 2L), d_out = 5L,
      .args = small(attention_type = at))
    expect_class(net, "nn_saint")
    x_num = torch_randn(7, 3)
    x_cat = torch_stack(list(
      torch_randint(1, 4, 7, dtype = torch_long()),
      torch_randint(1, 2, 7, dtype = torch_long())
    ), dim = 2)
    out = net(x_num = x_num, x_cat = x_cat)
    expect_equal(out$shape, c(7, 5))
  }
})

test_that("nn_saint works with only numeric or only categorical features", {
  net = invoke(nn_saint, n_features_num = 4L, cardinalities = integer(0), d_out = 2L, .args = small())
  expect_equal(net(x_num = torch_randn(3, 4))$shape, c(3, 2))

  net = invoke(nn_saint, n_features_num = 0L, cardinalities = c(3L, 3L), d_out = 2L, .args = small())
  x_cat = torch_ones(3, 2, dtype = torch_long())
  expect_equal(net(x_cat = x_cat)$shape, c(3, 2))
  # mlr3torch calls the network by position when there is a single input tensor
  expect_equal(net(x_cat)$shape, c(3, 2))

  expect_error(nn_saint(n_features_num = 0L, cardinalities = integer(0), d_out = 2L),
    "at least one feature")
})

test_that("nn_saint can be constructed from a task", {
  task = tsk("german_credit")
  net = invoke(nn_saint, task = task, .args = small())
  expect_equal(net$n_features_num, n_num_features(task))
  expect_equal(net$n_features_categ, n_categ_features(task))
  expect_equal(net$head$linear2$out_features, output_dim_for(task))
})

test_that("cardinalities are derived in the column order of the ingress token", {
  task = tsk("german_credit")
  info = saint_categ_info(task)
  sel = selector_type(c("factor", "ordered", "logical"))
  expect_equal(unname(info$cardinalities), unname(lengths(task$levels(sel(task)))))
  expect_true(all(info$min_code == 1L))
})

test_that("logical features get cardinality 2 and 0-based codes", {
  task = as_task_classif(data.frame(
    y = factor(rep(c("a", "b"), 10)), l = rep(c(TRUE, FALSE), 10), n = rnorm(20)
  ), target = "y")
  info = saint_categ_info(task)
  expect_equal(unname(info$cardinalities), 2L)
  expect_equal(unname(info$min_code), 0L)

  learner = invoke(lrn, "classif.saint", epochs = 1L, batch_size = 10L, .args = small())
  expect_error(learner$train(task), regexp = NA)
  expect_prediction(learner$predict(task))
})

test_that("LearnerTorchSAINT works on a multiclass task", {
  learner = invoke(lrn, "classif.saint", epochs = 1L, batch_size = 16L, .args = small())
  task = tsk("iris")
  learner$train(task)
  expect_prediction(learner$predict(task))
  learner$predict_type = "prob"
  pred = learner$predict(task)
  expect_prediction(pred)
  expect_matrix(pred$prob, ncols = 3L, nrows = task$nrow)
  expect_learner_torch(learner, task = task)
})

test_that("LearnerTorchSAINT works on a binary, numeric-only task", {
  learner = invoke(lrn, "classif.saint", epochs = 1L, batch_size = 16L, .args = small())
  expect_learner_torch(learner, tsk("sonar"))
})

test_that("LearnerTorchSAINT works on a mixed numeric + categorical task", {
  learner = invoke(lrn, "classif.saint", epochs = 1L, batch_size = 16L, .args = small())
  task = tsk("german_credit")$filter(1:60)
  expect_learner_torch(learner, task)
})

test_that("LearnerTorchSAINT works for regression", {
  learner = invoke(lrn, "regr.saint", epochs = 1L, batch_size = 16L, .args = small())
  task = tsk("mtcars")
  learner$train(task)
  expect_prediction(learner$predict(task))
  expect_learner_torch(learner, task)
})

test_that("all attention types can be trained", {
  for (at in c("col", "row", "colrow")) {
    learner = invoke(lrn, "classif.saint", epochs = 1L, batch_size = 16L,
      .args = small(attention_type = at))
    expect_error(learner$train(tsk("iris")), regexp = NA)
    expect_prediction(learner$predict(tsk("iris")))
  }
})

test_that("cloning also keeps parameter values", {
  learner = lrn("classif.saint", depth = 2L)
  learnerc = learner$clone(deep = TRUE)
  expect_deep_clone_mlr3torch(learner, learnerc)
  expect_equal(learner$param_set$values$depth, 2L)
  expect_equal(learnerc$param_set$values$depth, 2L)
})

test_that("intersample attention makes predictions batch-dependent", {
  # This test *documents* the behaviour described in the 'Intersample Attention' section of the
  # learner: with row attention the prediction for an observation depends on which other
  # observations are in the same batch. This is inherent to SAINT, not a bug.
  task = tsk("iris")

  get_prob = function(at, bs) {
    learner = invoke(lrn, "classif.saint", epochs = 1L, batch_size = 30L, seed = 1L,
      predict_type = "prob", .args = small(attention_type = at))
    learner$train(task)
    learner$param_set$set_values(batch_size = bs)
    learner$predict(task)$prob
  }

  # column attention only: predictions do not depend on the batch composition
  # (up to the float32 non-determinism of differently sized matmuls)
  expect_equal(get_prob("col", 30L), get_prob("col", 50L), tolerance = 1e-5)

  # with intersample attention they do
  p1 = get_prob("colrow", 30L)
  p2 = get_prob("colrow", 50L)
  expect_false(isTRUE(all.equal(p1, p2, tolerance = 1e-4)))
})

test_that("a small predict batch_size warns when intersample attention is active", {
  learner = invoke(lrn, "classif.saint", epochs = 1L, batch_size = 16L, .args = small())
  task = tsk("iris")
  learner$train(task)
  learner$param_set$set_values(batch_size = 2L)
  expect_warning(learner$predict(task), "intersample attention")

  learner2 = invoke(lrn, "classif.saint", epochs = 1L, batch_size = 2L,
    .args = small(attention_type = "col"))
  learner2$train(task)
  expect_warning(learner2$predict(task), regexp = NA)
})

test_that("informative errors for unsupported input", {
  learner = invoke(lrn, "classif.saint", epochs = 1L, batch_size = 16L, .args = small())
  expect_error(learner$train(tsk("lazy_iris")), "lazy_tensor")
  expect_error(nn_saint(n_features_num = 1L, cardinalities = 1L, d_out = 1L,
    attention_type = "rowcol"), "attention_type")
})
