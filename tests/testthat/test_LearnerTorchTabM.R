make_cat = function(n, cardinalities) {
  torch_stack(lapply(cardinalities, function(card) {
    torch_randint(1L, card, n, dtype = torch_long())
  }), dim = 2L)
}

# --------------------------------------------------------------------------------------
# Embeddings for numerical features (the `rtdl_num_embeddings` port)
# --------------------------------------------------------------------------------------

test_that("nn_linear_embeddings works", {
  m = nn_linear_embeddings(3L, 4L)
  expect_class(m, "nn_linear_embeddings")
  expect_equal(m$get_output_shape(), c(3L, 4L))
  expect_equal(m(torch_randn(5, 3))$shape, c(5, 3, 4))
  # each feature is embedded by its own affine map
  x = torch_randn(2, 3)
  manual = x$unsqueeze(-1L) * m$weight + m$bias
  expect_true(as.logical((torch_abs(m(x) - manual) < 1e-6)$all()))
  expect_error(m(torch_randn(5, 2)), "last dimension of the input")
})

test_that("nn_linear_relu_embeddings works", {
  m = nn_linear_relu_embeddings(3L, 8L)
  expect_equal(m$get_output_shape(), c(3L, 8L))
  out = m(torch_randn(5, 3))
  expect_equal(out$shape, c(5, 3, 8))
  expect_true(as.logical((out >= 0)$all()))
  # the upstream default embedding size
  expect_equal(nn_linear_relu_embeddings(3L)$get_output_shape(), c(3L, 32L))
})

test_that("nn_periodic_embeddings works", {
  m = nn_periodic_embeddings(3L, 8L, n_frequencies = 5L, lite = FALSE)
  expect_equal(m$get_output_shape(), c(3L, 8L))
  expect_equal(m(torch_randn(5, 3))$shape, c(5, 3, 8))
  # without `lite`, every feature has its own outer linear layer
  expect_class(m$linear, "nn_nlinear")
  expect_equal(m$linear$weight$shape, c(3, 10, 8))

  m = nn_periodic_embeddings(3L, 8L, n_frequencies = 5L, lite = TRUE)
  expect_class(m$linear, "nn_linear")
  expect_equal(m(torch_randn(5, 3))$shape, c(5, 3, 8))

  torch_manual_seed(1)
  m = nn_periodic_embeddings(3L, 8L, n_frequencies = 5L, activation = FALSE, lite = FALSE)
  expect_null(m$activation)
  expect_true(as.logical((m(torch_randn(50, 3)) < 0)$any()))

  expect_error(nn_periodic_embeddings(3L, 8L, activation = FALSE, lite = TRUE),
    "lite = TRUE is allowed only when activation = TRUE")
  # the upstream defaults
  expect_equal(nn_periodic_embeddings(3L, lite = FALSE)$get_output_shape(), c(3L, 24L))
  expect_equal(nn_periodic_embeddings(3L, lite = FALSE)$periodic$weight$shape, c(3, 48))
})

test_that("the frequencies are initialized from a truncated normal", {
  torch_manual_seed(1)
  sigma = 0.05
  m = nn_periodic_embeddings(500L, 4L, n_frequencies = 200L, frequency_init_scale = sigma,
    lite = TRUE)
  w = m$periodic$weight$detach()
  expect_true(as.logical((torch_abs(w) <= 3 * sigma)$all()))
  # the sd of N(0, sigma) truncated at +- 3 sigma
  expected_sd = sigma * sqrt(1 - 2 * 3 * dnorm(3) / (2 * pnorm(3) - 1))
  expect_equal(as.numeric(w$std()), expected_sd, tolerance = 0.02)
  expect_equal(as.numeric(w$mean()), 0, tolerance = 0.01)
})

test_that("compute_bins works", {
  withr::local_seed(1L)
  x = matrix(rnorm(1000), ncol = 5L)
  bins = compute_bins(x, n_bins = 10L)
  expect_list(bins, types = "torch_tensor", len = 5L)
  for (b in bins) {
    expect_equal(b$dim(), 1L)
    expect_true(b$shape[1L] <= 11L)
    # sorted and within the data range
    expect_true(as.logical((torch_diff(b) > 0)$all()))
  }
  # the outer bin edges are the minimum and the maximum
  expect_equal(as.numeric(bins[[1L]][1L]), min(x[, 1L]), tolerance = 1e-6)
  expect_equal(as.numeric(bins[[1L]][bins[[1L]]$shape[1L]]), max(x[, 1L]), tolerance = 1e-6)

  # coinciding quantiles are removed, so a feature can end up with fewer bins
  x = cbind(rnorm(100), rep(c(0, 1), 50L))
  bins = compute_bins(x, n_bins = 10L)
  expect_equal(bins[[1L]]$shape[1L], 11L)
  # only the edges 0, 0.5 and 1 remain for the binary column
  expect_equal(bins[[2L]]$shape[1L], 3L)
  expect_equal(as.numeric(bins[[2L]]), c(0, 0.5, 1), tolerance = 1e-6)

  expect_error(compute_bins(matrix(1, nrow = 10L, ncol = 2L), n_bins = 5L),
    "at least two distinct values")
  expect_error(compute_bins(matrix(rnorm(20), ncol = 2L), n_bins = 20L),
    "smaller than the number of rows")
  expect_error(compute_bins(matrix(c(NA_real_, rnorm(19)), ncol = 2L), n_bins = 5L),
    "must not contain NaN")
})

test_that("nn_piecewise_linear_embeddings works", {
  bins = compute_bins(matrix(rnorm(1000), ncol = 5L), n_bins = 10L)
  m = nn_piecewise_linear_embeddings(bins, d_embedding = 8L, activation = FALSE,
    version = "B")
  expect_equal(m$get_output_shape(), c(5L, 8L))
  expect_equal(m(torch_randn(6, 5))$shape, c(6, 5, 8))
  # version "B" adds the linear shortcut and zero-initializes the piecewise-linear part,
  # so the module starts out as a plain linear embedding
  expect_class(m$linear0, "nn_linear_embeddings")
  expect_true(as.logical((m$linear$weight == 0)$all()))
  x = torch_randn(4, 5)
  expect_true(as.logical((torch_abs(m(x) - m$linear0(x)) < 1e-6)$all()))

  m = nn_piecewise_linear_embeddings(bins, d_embedding = 8L, activation = TRUE,
    version = "A")
  expect_null(m$linear0)
  out = m(torch_randn(6, 5))
  expect_equal(out$shape, c(6, 5, 8))
  expect_true(as.logical((out >= 0)$all()))

  expect_error(nn_piecewise_linear_embeddings(bins, 8L, activation = TRUE, version = "C"),
    "version")
  expect_error(nn_piecewise_linear_embeddings(bins, 8L, activation = TRUE),
    "argument \"version\" is missing")
  expect_error(m(torch_randn(2, 6, 5)), "exactly one batch dimension")
})

test_that("the piecewise-linear encoding is correct", {
  # one feature with the bin edges 0, 1, 3 (i.e. two bins of width 1 and 2)
  bins = list(torch_tensor(c(0, 1, 3)))
  impl = nn_piecewise_linear_encoding_impl(bins)
  expect_equal(impl$get_max_n_bins(), 2L)
  x = torch_tensor(matrix(c(-1, 0, 0.5, 1, 2, 3, 4), ncol = 1L))
  out = as.matrix(impl(x)$squeeze(2L))
  # first component: min(x / 1, 1); second: max((x - 1) / 2, 0)
  expect_equal(out[, 1L], pmin(as.numeric(x), 1), tolerance = 1e-6)
  expect_equal(out[, 2L], pmax((as.numeric(x) - 1) / 2, 0), tolerance = 1e-6)

  # a feature with a single bin behaves like min-max scaling (no lower clamp)
  bins = list(torch_tensor(c(0, 1, 3)), torch_tensor(c(0, 2)))
  impl = nn_piecewise_linear_encoding_impl(bins)
  expect_false(is.null(impl$single_bin_mask))
  out = impl(torch_tensor(matrix(c(0.5, -1), nrow = 1L)))
  expect_equal(as.numeric(out[1, 2, 2]), -0.5, tolerance = 1e-6)
})

test_that("all embedding modules can be jit traced", {
  bins = compute_bins(matrix(rnorm(400), ncol = 4L), n_bins = 5L)
  modules = list(
    nn_linear_embeddings(4L, 6L),
    nn_linear_relu_embeddings(4L, 6L),
    nn_periodic_embeddings(4L, 6L, n_frequencies = 5L, lite = FALSE),
    nn_periodic_embeddings(4L, 6L, n_frequencies = 5L, lite = TRUE),
    nn_piecewise_linear_embeddings(bins, 6L, activation = FALSE, version = "B")
  )
  for (m in modules) {
    m$eval()
    traced = jit_trace(m, torch_randn(5, 4))
    # a different batch size must still work
    expect_equal(traced(torch_randn(3, 4))$shape, c(3, 4, 6))
    expect_true(as.logical((torch_abs(traced(torch_randn(3, 4)) - m(torch_randn(3, 4))) < 1e6)$all()))
  }
})

test_that("gradients reach every parameter of the embedding modules", {
  bins = compute_bins(matrix(rnorm(400), ncol = 4L), n_bins = 5L)
  modules = list(
    nn_linear_embeddings(4L, 6L),
    nn_linear_relu_embeddings(4L, 6L),
    nn_periodic_embeddings(4L, 6L, n_frequencies = 5L, lite = FALSE),
    nn_piecewise_linear_embeddings(bins, 6L, activation = FALSE, version = "A")
  )
  for (m in modules) {
    m(torch_randn(8, 4))$sum()$backward()
    for (p in m$parameters) {
      expect_false(is_undefined_tensor(p$grad))
      expect_true(as.logical((p$grad != 0)$any()))
    }
  }
})

# --------------------------------------------------------------------------------------
# TabM
# --------------------------------------------------------------------------------------

test_that("nn_tabm_one_hot checks the number of categorical features", {
  # upstream raises here; without the check the surplus columns would be dropped silently,
  # because only the first `length(cardinalities)` are read
  oh = nn_tabm_one_hot(c(3L, 2L))
  correct = torch_tensor(matrix(c(1L, 2L, 2L, 1L), nrow = 2L, byrow = TRUE), dtype = torch_long())
  expect_equal(oh(correct)$shape, c(2L, 5L))

  too_many = torch_tensor(matrix(c(1L, 2L, 3L, 2L, 1L, 1L), nrow = 2L, byrow = TRUE),
    dtype = torch_long())
  expect_error(oh(too_many), "Expected 2 categorical features, but got 3")
  too_few = torch_tensor(matrix(c(1L, 2L), ncol = 1L), dtype = torch_long())
  expect_error(oh(too_few), "Expected 2 categorical features, but got 1")
})

test_that("nn_tabm works for all architecture types", {
  for (arch_type in c("tabm", "tabm-mini", "tabm-packed")) {
    net = nn_tabm(n_num_features = 4L, cat_cardinalities = c(3L, 2L), d_out = 3L,
      arch_type = arch_type, k = 5L, n_blocks = 2L, d_block = 8L, dropout = 0.1)
    expect_class(net, "nn_tabm")
    out = net(x_num = torch_randn(7, 4), x_cat = make_cat(7, c(3L, 2L)))
    # (batch, k, d_out)
    expect_equal(out$shape, c(7, 5, 3))
  }
  # the "plain" arch_type of paper/bin/model.py is not part of the packaged tabm.py
  expect_error(nn_tabm(n_num_features = 4L, d_out = 1L, arch_type = "plain"), "arch_type")
})

test_that("nn_tabm works with only numeric or only categorical features", {
  net = nn_tabm(n_num_features = 4L, d_out = 2L, k = 3L, n_blocks = 1L, d_block = 6L,
    dropout = 0)
  expect_equal(net(x_num = torch_randn(5, 4))$shape, c(5, 3, 2))

  net = nn_tabm(cat_cardinalities = c(3L, 2L), d_out = 2L, k = 3L, n_blocks = 1L,
    d_block = 6L, dropout = 0)
  expect_equal(net(x_cat = make_cat(5, c(3L, 2L)))$shape, c(5, 3, 2))
})

test_that("nn_tabm can be constructed from a task", {
  task = tsk("german_credit")
  net = nn_tabm(task = task, k = 2L, n_blocks = 1L, d_block = 4L, dropout = 0)
  expect_equal(net$n_num_features, n_num_features(task))
  expect_equal(net$n_cat_features, n_categ_features(task))
  expect_equal(net$output$out_features, output_dim_for(task))
})

test_that("nn_tabm gives informative errors", {
  expect_error(nn_tabm(n_num_features = 0L, d_out = 1L),
    "at least one numerical or one categorical feature")
  expect_error(nn_tabm(n_num_features = 2L, d_out = 1L, arch_type = "wrong"),
    "arch_type")
  expect_error(
    nn_tabm(n_num_features = 2L, d_out = 1L, arch_type = "tabm-packed",
      start_scaling_init = "normal"),
    "start_scaling_init must be NULL")
  net = nn_tabm(n_num_features = 2L, cat_cardinalities = 3L, d_out = 1L, k = 2L,
    n_blocks = 1L, d_block = 4L, dropout = 0)
  expect_error(net(x_num = torch_randn(3, 2)), "x_cat is NULL")
  expect_error(net(x_cat = make_cat(3, 3L)), "x_num is NULL")
})

test_that("categorical features incl. logicals are handled correctly", {
  # TabM used to carry its own cardinality helper and batchgetter because the generic
  # `batchgetter_categ()` encoded logicals as 0/1. Both now come from mlr3torch itself.
  dat = data.frame(y = factor(c("a", "b", "a", "b")), l = c(TRUE, FALSE, TRUE, TRUE),
    f = factor(c("x", "y", "x", "z")), n = c(1, 2, 3, 4))
  task_l = as_task_classif(dat, target = "y", id = "logi")
  expect_equal(unname(categ_cardinalities(task_l)), c(3L, 2L))

  codes = batchgetter_categ(task_l$data(cols = ingress_categ()$features(task_l)))
  expect_true(as.logical((codes >= 1L)$all()))

  # and the learner trains on a task that mixes factors, logicals and numerics
  learner = lrn("classif.tabm", epochs = 1L, batch_size = 4L, k = 2L, n_blocks = 1L,
    d_block = 8L, predict_type = "prob")
  learner$train(task_l)
  expect_prediction(learner$predict(task_l))
})

test_that("the learner works on tasks without numerical features", {
  # When a task has a single input tensor, `learner_torch_train()` calls the network by
  # *position*, so a purely categorical task arrives in nn_tabm's first formal, `x_num`.
  # The module-level tests above pass `x_cat` by name and therefore cannot catch this.
  tabm = function() {
    lrn("classif.tabm", epochs = 1L, batch_size = 10L, k = 2L, n_blocks = 1L, d_block = 8L)
  }
  n = 20L

  categ_only = as_task_classif(data.frame(
    y = factor(rep(c("a", "b"), n / 2)),
    f1 = factor(rep(c("u", "v", "w"), length.out = n)),
    f2 = factor(rep(c("p", "q"), length.out = n))
  ), target = "y")
  learner = tabm()
  expect_error(learner$train(categ_only), regexp = NA)
  expect_prediction(learner$predict(categ_only))

  logical_only = as_task_classif(data.frame(
    y = factor(rep(c("a", "b"), n / 2)),
    l1 = rep(c(TRUE, FALSE), n / 2),
    l2 = rep(c(TRUE, TRUE, FALSE, FALSE), n / 4)
  ), target = "y")
  learner = tabm()
  expect_error(learner$train(logical_only), regexp = NA)
  expect_prediction(learner$predict(logical_only))

  # the informative error is still raised when a network that *does* expect numerical features
  # gets none
  net = nn_tabm(n_num_features = 2L, cat_cardinalities = integer(0), d_out = 2L, k = 2L,
    n_blocks = 1L, d_block = 8L)
  expect_error(net(x_cat = NULL), "x_num is NULL")
})

test_that("LearnerTorchTabM works for all architecture types", {
  task = tsk("iris")
  for (arch_type in c("tabm", "tabm-mini", "tabm-packed")) {
    learner = lrn("classif.tabm", epochs = 1L, batch_size = 50L, arch_type = arch_type,
      k = 3L, n_blocks = 1L, d_block = 8L, predict_type = "prob")
    learner$train(task)
    pred = learner$predict(task)
    expect_prediction(pred)
    expect_true(all(abs(rowSums(pred$prob) - 1) < 1e-5))
  }
})

test_that("expect_learner_torch on multiclass, binary, mixed and regression tasks", {
  learner = lrn("classif.tabm", epochs = 1L, batch_size = 16L, k = 3L, n_blocks = 1L,
    d_block = 8L)
  expect_learner_torch(learner, tsk("iris"), k = 3L)
  expect_learner_torch(learner, tsk("sonar"), k = 3L)
  expect_learner_torch(learner, tsk("german_credit")$filter(1:100), k = 3L)

  learner = lrn("regr.tabm", epochs = 1L, batch_size = 16L, k = 3L, n_blocks = 1L,
    d_block = 8L)
  expect_learner_torch(learner, tsk("mtcars"), k = 3L)
})

test_that("predicted probabilities are the mean of the per-submodel probabilities", {
  # multiclass
  learner = lrn("classif.tabm", epochs = 1L, batch_size = 50L, k = 4L, n_blocks = 1L,
    d_block = 8L, predict_type = "prob", seed = 1L)
  task = tsk("iris")
  learner$train(task)
  pred = learner$predict(task)

  network = learner$model$network
  network$eval()
  batch = learner$dataset(task)$.getbatch(seq_len(task$nrow))
  raw = with_no_grad(invoke(network$forward, .args = batch$x))
  expect_equal(raw$shape, c(task$nrow, 4L, 3L))
  # averaging probabilities, *not* logits
  manual = as.matrix(nnf_softmax(raw, dim = 3L)$mean(dim = 2L))
  colnames(manual) = task$class_names
  expect_lt(max(abs(pred$prob - manual)), 1e-6)
  expect_false(isTRUE(all.equal(pred$prob, as.matrix(nnf_softmax(raw$mean(dim = 2L), dim = 2L)),
    check.attributes = FALSE)))

  # binary
  learner = lrn("classif.tabm", epochs = 1L, batch_size = 50L, k = 4L, n_blocks = 1L,
    d_block = 8L, predict_type = "prob", seed = 1L)
  task = tsk("sonar")
  learner$train(task)
  pred = learner$predict(task)
  network = learner$model$network
  network$eval()
  batch = learner$dataset(task)$.getbatch(seq_len(task$nrow))
  raw = with_no_grad(invoke(network$forward, .args = batch$x))
  manual = as.numeric(torch_sigmoid(raw)$mean(dim = 2L))
  expect_lt(max(abs(unname(pred$prob[, task$class_names[1L]]) - manual)), 1e-6)
  expect_lt(max(abs(unname(pred$prob[, task$class_names[2L]]) - (1 - manual))), 1e-6)

  # regression: the responses are the mean over the k submodels
  learner = lrn("regr.tabm", epochs = 1L, batch_size = 16L, k = 4L, n_blocks = 1L,
    d_block = 8L, seed = 1L)
  task = tsk("mtcars")
  learner$train(task)
  pred = learner$predict(task)
  network = learner$model$network
  network$eval()
  batch = learner$dataset(task)$.getbatch(seq_len(task$nrow))
  raw = with_no_grad(invoke(network$forward, .args = batch$x))
  expect_equal(pred$response, as.numeric(raw$mean(dim = 2L)), tolerance = 1e-6)
})

test_that("the loss folds the ensemble dimension into the batch dimension", {
  # multiclass
  loss_fn = tabm_wrap_loss(t_loss("cross_entropy"))$generate(tsk("iris"))
  expect_class(loss_fn, "nn_tabm_loss")
  input = torch_randn(6, 4, 3)
  target = torch_randint(1L, 3L, 6, dtype = torch_long())
  expect_equal(
    as.numeric(loss_fn(input, target)),
    as.numeric(nnf_cross_entropy(input$flatten(start_dim = 1L, end_dim = 2L),
      target$repeat_interleave(4L, dim = 1L))),
    tolerance = 1e-6
  )

  # binary
  loss_fn = tabm_wrap_loss(t_loss("cross_entropy"))$generate(tsk("sonar"))
  input = torch_randn(6, 4, 1)
  target = torch_randint(0L, 1L, 6, dtype = torch_long())$to(dtype = torch_float())$unsqueeze(2L)
  expect_equal(
    as.numeric(loss_fn(input, target)),
    as.numeric(nnf_binary_cross_entropy_with_logits(
      input$flatten(start_dim = 1L, end_dim = 2L), target$repeat_interleave(4L, dim = 1L))),
    tolerance = 1e-6
  )

  # regression
  loss_fn = tabm_wrap_loss(t_loss("mse"))$generate(tsk("mtcars"))
  input = torch_randn(6, 4, 1)
  target = torch_randn(6, 1)
  expect_equal(
    as.numeric(loss_fn(input, target)),
    as.numeric(nnf_mse_loss(input$flatten(start_dim = 1L, end_dim = 2L),
      target$repeat_interleave(4L, dim = 1L))),
    tolerance = 1e-6
  )
})

test_that("a user-supplied loss is wrapped as well", {
  learner = lrn("regr.tabm", loss = t_loss("l1"), epochs = 1L, batch_size = 16L, k = 3L,
    n_blocks = 1L, d_block = 8L)
  expect_equal(learner$loss$id, "l1")
  expect_class(learner$loss$generate(tsk("mtcars")), "nn_tabm_loss")
  expect_error(learner$train(tsk("mtcars")), regexp = NA)

  # assignment after construction is wrapped, too, and wrapping is idempotent
  learner$loss = t_loss("mse")
  expect_class(learner$loss$generate(tsk("mtcars")), "nn_tabm_loss")
  learner$loss = learner$loss
  expect_class(learner$loss$generate(tsk("mtcars")), "nn_tabm_loss")
})

test_that("nn_tabm can be used with lrn('classif.module')", {
  loss = TorchLoss$new(nn_tabm_loss, task_types = "classif")
  loss$param_set$set_values(loss = nn_cross_entropy_loss())
  learner = lrn("classif.module",
    module_generator = nn_tabm,
    ingress_tokens = list(x_num = ingress_num()),
    loss = loss,
    epochs = 1L, batch_size = 50L, k = 3L, n_blocks = 1L, d_block = 8L
  )
  # the module infers n_num_features / cat_cardinalities / d_out from the task
  expect_error(learner$train(tsk("iris")), regexp = NA)
  expect_equal(learner$model$network$k, 3L)

  # Training works, predicting does not: the default prediction encoder expects a
  # (batch, d_out) output and reads the ensemble dimension as additional observations.
  # Aggregating over the k members needs a custom prediction encoder, see the
  # 'Ensemble Output' section of `nn_tabm`. `LearnerTorchTabM` does this itself.
  expect_error(learner$predict(tsk("iris")), "additional predictions")
})

test_that("cloning also keeps parameter values", {
  learner = lrn("classif.tabm", n_blocks = 2L, k = 8L)
  learnerc = learner$clone(deep = TRUE)
  expect_deep_clone_mlr3torch(learner, learnerc)
  expect_equal(learner$param_set$values$n_blocks, 2L)
  expect_equal(learnerc$param_set$values$n_blocks, 2L)
  expect_equal(learnerc$param_set$values$k, 8L)
})

test_that("jit_trace works", {
  learner = lrn("classif.tabm", epochs = 1L, batch_size = 64L, k = 3L, n_blocks = 1L,
    d_block = 8L, jit_trace = TRUE, predict_type = "prob")
  task = tsk("german_credit")$filter(1:100)
  learner$train(task)
  expect_prediction(learner$predict(task))
})

test_that("informative errors for unsupported input", {
  learner = lrn("classif.tabm", epochs = 1L, batch_size = 16L, k = 2L, n_blocks = 1L,
    d_block = 8L)
  # lazy tensors are not supported by this learner
  expect_error(learner$train(tsk("lazy_iris")), "lazy_tensor")
  expect_error(learner$train(nano_imagenet()), "lazy_tensor")
})

test_that("nn_tabm works with numerical feature embeddings", {
  bins = compute_bins(matrix(rnorm(400), ncol = 4L), n_bins = 5L)
  embeddings = list(
    linear_relu = nn_linear_relu_embeddings(4L, 6L),
    periodic = nn_periodic_embeddings(4L, 6L, lite = FALSE),
    periodic_lite = nn_periodic_embeddings(4L, 6L, lite = TRUE),
    piecewise_linear = nn_piecewise_linear_embeddings(bins, 6L, activation = FALSE, version = "B")
  )
  for (nm in names(embeddings)) {
    net = nn_tabm(n_num_features = 4L, cat_cardinalities = c(3L, 2L), d_out = 3L,
      num_embeddings = embeddings[[nm]], k = 5L, d_block = 8L)
    out = net(x_num = torch_randn(7, 4), x_cat = make_cat(7, c(3L, 2L)))
    expect_equal(out$shape, c(7, 5, 3), info = nm)
    # upstream `TabM.make()`: 2 blocks with embeddings, and the flat input size is
    # n_num * d_embedding + sum(cardinalities)
    expect_equal(net$backbone$n_blocks, 2L, info = nm)
    expect_equal(net$backbone$blocks[[1L]][[1L]]$in_features, 4L * 6L + 5L, info = nm)
  }
  # ... and 3 blocks without
  net = nn_tabm(n_num_features = 4L, d_out = 3L, k = 2L, d_block = 8L)
  expect_equal(net$backbone$n_blocks, 3L)
})

test_that("nn_tabm validates num_embeddings", {
  bins = compute_bins(matrix(rnorm(400), ncol = 4L), n_bins = 5L)
  expect_error(
    nn_tabm(cat_cardinalities = 3L, d_out = 1L,
      num_embeddings = nn_linear_relu_embeddings(4L, 6L)),
    "no numerical features")
  expect_error(
    nn_tabm(n_num_features = 3L, d_out = 1L,
      num_embeddings = nn_linear_relu_embeddings(4L, 6L)),
    "created for 4 features")
  # TabM requires version "B" of the piecewise-linear embeddings
  expect_error(
    nn_tabm(n_num_features = 4L, d_out = 1L,
      num_embeddings = nn_piecewise_linear_embeddings(bins, 6L, activation = TRUE, version = "A")),
    "version = \"B\"")
  expect_error(nn_tabm(n_num_features = 4L, d_out = 1L, num_embeddings = nn_relu()),
    "get_output_shape")
})

test_that("nn_tabm applies upstream's start_scaling_init heuristic", {
  # "random-signs" without embeddings, "normal" with embeddings
  net = nn_tabm(n_num_features = 4L, d_out = 1L, k = 3L, n_blocks = 1L, d_block = 4L)
  expect_equal(net$backbone$blocks[[1L]][[1L]]$first_scaling_init, "random-signs")
  net = nn_tabm(n_num_features = 4L, d_out = 1L, k = 3L, n_blocks = 1L, d_block = 4L,
    num_embeddings = nn_linear_relu_embeddings(4L, 6L))
  expect_equal(net$backbone$blocks[[1L]][[1L]]$first_scaling_init, "normal")
  # explicitly given values win
  net = nn_tabm(n_num_features = 4L, d_out = 1L, k = 3L, n_blocks = 1L, d_block = 4L,
    start_scaling_init = "normal")
  expect_equal(net$backbone$blocks[[1L]][[1L]]$first_scaling_init, "normal")
})

test_that("LearnerTorchTabM works with all num_embeddings types", {
  task = tsk("iris")
  for (emb in c("none", "linear_relu", "periodic", "piecewise_linear")) {
    learner = lrn("classif.tabm", epochs = 1L, batch_size = 50L, k = 3L, d_block = 8L,
      num_embeddings = emb, d_embedding = 4L, n_frequencies = 4L, n_bins = 5L,
      predict_type = "prob")
    learner$train(task)
    expect_prediction(learner$predict(task))
    net = learner$model$network
    expect_equal(net$backbone$n_blocks, if (emb == "none") 3L else 2L, info = emb)
    expect_equal(net$backbone$blocks[[1L]][[1L]]$first_scaling_init,
      if (emb == "none") "random-signs" else "normal", info = emb)
    expect_equal(is.null(net$num_module), emb == "none", info = emb)
  }
  # regression, and `lite` periodic embeddings
  learner = lrn("regr.tabm", epochs = 1L, batch_size = 16L, k = 3L, n_blocks = 1L,
    d_block = 8L, num_embeddings = "periodic", d_embedding = 4L, n_frequencies = 4L,
    lite = TRUE)
  learner$train(tsk("mtcars"))
  expect_prediction(learner$predict(tsk("mtcars")))
})

test_that("the bins of the piecewise-linear embeddings come from the training data", {
  withr::local_seed(1L)
  task = tsk("iris")
  learner = lrn("classif.tabm", epochs = 1L, batch_size = 50L, k = 2L, d_block = 8L,
    num_embeddings = "piecewise_linear", d_embedding = 4L, n_bins = 5L)
  ids = partition(task, ratio = 0.5)
  learner$train(task, row_ids = ids$train)
  num_features = task$feature_names[task$feature_types$type %in% c("numeric", "integer")]
  expected = compute_bins(batchgetter_num(task$data(ids$train, cols = num_features)), n_bins = 5L)
  impl = learner$model$network$num_module$impl
  reference = nn_piecewise_linear_encoding_impl(expected)
  expect_true(torch_equal(impl$weight, reference$weight))
  expect_true(torch_equal(impl$bias, reference$bias))
})

test_that("LearnerTorchTabM gives an informative error when n_bins is too large", {
  learner = lrn("regr.tabm", epochs = 1L, batch_size = 8L, k = 2L, n_blocks = 1L,
    d_block = 8L, num_embeddings = "piecewise_linear", n_bins = 48L)
  expect_error(learner$train(tsk("mtcars")),
    "Cannot compute the bins for the piecewise-linear embeddings")
  learner$param_set$set_values(num_embeddings = "periodic")
  expect_error(learner$train(tsk("mtcars")), regexp = NA)
})

test_that("the activation can be a name, a module generator or a function", {
  activation_class = function(activation) {
    net = nn_tabm(n_num_features = 4L, d_out = 1L, k = 2L, n_blocks = 1L, d_block = 4L,
      activation = activation)
    class(net$backbone$blocks[[1L]][[2L]])[[1L]]
  }
  expect_equal(activation_class("relu"), "nn_relu")
  expect_equal(activation_class("nn_gelu"), "nn_gelu")
  # the torch.nn spelling used by upstream
  expect_equal(activation_class("ReLU"), "nn_relu")
  expect_equal(activation_class("LeakyReLU"), "nn_leaky_relu")
  expect_equal(activation_class(nn_tanh), "nn_tanh")
  expect_equal(activation_class(function() nn_elu(alpha = 2)), "nn_elu")

  expect_error(activation_class("does_not_exist"), "Cannot resolve the activation")
  expect_error(activation_class(1L), "must be a `character\\(1\\)`")
  expect_error(activation_class(function() 42), "must return an `nn_module`")
})

test_that("a non-default activation works in the learner and affects the hash", {
  learner = lrn("classif.tabm", epochs = 1L, batch_size = 50L, k = 2L, n_blocks = 1L,
    d_block = 8L, activation = nn_gelu)
  learner$train(tsk("iris"))
  expect_class(learner$model$network$backbone$blocks[[1L]][[2L]], "nn_gelu")
  expect_prediction(learner$predict(tsk("iris")))

  l1 = lrn("classif.tabm", activation = "relu")
  l2 = lrn("classif.tabm", activation = nn_gelu)
  expect_true(l1$hash != l2$hash)
  # the parameter *values* are not part of the phash
  expect_true(l1$phash == l2$phash)
  l2c = l2$clone(deep = TRUE)
  expect_deep_clone_mlr3torch(l2, l2c)
  expect_true(identical(l2c$param_set$values$activation, nn_gelu))

  # invalid values are rejected by the ParamSet
  expect_error(lrn("classif.tabm", activation = 1L), "activation")
})

test_that("the learner encodes all categorical features with 1-based codes", {
  dat = data.frame(y = factor(rep(c("a", "b"), 10L)), lg = rep(c(TRUE, FALSE), 10L),
    fc = factor(rep(c("x", "y", "z", "x"), 5L)), nm = rnorm(20))
  task = as_task_classif(dat, target = "y", id = "logi")
  learner = lrn("classif.tabm", epochs = 1L, batch_size = 10L, k = 2L, n_blocks = 1L,
    d_block = 8L)
  batch = learner$dataset(task)$.getbatch(1:5)
  expect_true(as.logical((batch$x$x_cat >= 1L)$all()))
  expect_true(as.logical((batch$x$x_cat <= torch_tensor(c(3L, 2L)))$all()))
  learner$train(task)
  expect_prediction(learner$predict(task))
})
