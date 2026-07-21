make_cat = function(n, cardinalities) {
  torch_stack(lapply(cardinalities, function(card) {
    torch_randint(1L, card, n, dtype = torch_long())
  }), dim = 2L)
}

test_that("nn_tabm works for all architecture types", {
  for (arch_type in c("tabm", "tabm-mini", "tabm-packed", "plain")) {
    net = nn_tabm(n_num_features = 4L, cat_cardinalities = c(3L, 2L), d_out = 3L,
      arch_type = arch_type, k = 5L, n_blocks = 2L, d_block = 8L, dropout = 0.1)
    expect_class(net, "nn_tabm")
    out = net(x_num = torch_randn(7, 4), x_cat = make_cat(7, c(3L, 2L)))
    # (batch, k, d_out); "plain" is not ensembled and always has k = 1
    expect_equal(out$shape, c(7, if (arch_type == "plain") 1 else 5, 3))
  }
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

test_that("categorical cardinalities are in the ingress column order", {
  task = tsk("german_credit")
  categ = selector_type(c("factor", "ordered", "logical"))(task)
  expect_equal(unname(tabm_cardinalities(task)), unname(lengths(task$levels(categ))[categ]))

  # logical features have no levels in mlr3 and their codes are shifted by the batchgetter
  dat = data.frame(y = factor(c("a", "b", "a", "b")), l = c(TRUE, FALSE, TRUE, TRUE),
    f = factor(c("x", "y", "x", "z")), n = c(1, 2, 3, 4))
  task_l = as_task_classif(dat, target = "y", id = "logi")
  expect_equal(tabm_cardinalities(task_l), c(3L, 2L))
  codes = batchgetter_categ_tabm(task_l$data(cols = selector_type(c("factor", "ordered", "logical"))(task_l)))
  expect_true(as.logical((codes >= 1L)$all()))
})

test_that("LearnerTorchTabM works for all architecture types", {
  task = tsk("iris")
  for (arch_type in c("tabm", "tabm-mini", "tabm-packed", "plain")) {
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
    d_block = 8L, predict_type = "prob")
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
  expect_equal(pred$prob, manual, tolerance = 1e-6)
  expect_false(isTRUE(all.equal(pred$prob, as.matrix(nnf_softmax(raw$mean(dim = 2L), dim = 2L)),
    check.attributes = FALSE)))

  # binary
  learner = lrn("classif.tabm", epochs = 1L, batch_size = 50L, k = 4L, n_blocks = 1L,
    d_block = 8L, predict_type = "prob")
  task = tsk("sonar")
  learner$train(task)
  pred = learner$predict(task)
  network = learner$model$network
  network$eval()
  batch = learner$dataset(task)$.getbatch(seq_len(task$nrow))
  raw = with_no_grad(invoke(network$forward, .args = batch$x))
  manual = as.numeric(torch_sigmoid(raw)$mean(dim = 2L))
  expect_equal(unname(pred$prob[, task$class_names[1L]]), manual, tolerance = 1e-6)
  expect_equal(unname(pred$prob[, task$class_names[2L]]), 1 - manual, tolerance = 1e-6)

  # regression: the responses are the mean over the k submodels
  learner = lrn("regr.tabm", epochs = 1L, batch_size = 16L, k = 4L, n_blocks = 1L,
    d_block = 8L)
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
