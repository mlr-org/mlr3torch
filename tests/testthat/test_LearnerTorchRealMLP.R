make_realmlp = function(task_type = "classif", ...) {
  args = insert_named(list(epochs = 1L, batch_size = 16L, n_hidden_layers = 2L, hidden_width = 8L),
    list(...))
  invoke(lrn, paste0(task_type, ".realmlp"), .args = args)
}

wrap_realmlp_loss = function(learner, task) {
  realmlp_wrap_loss(learner$loss$generate(task), task, learner$loss,
    learner$param_set$get_values(tags = "train"), learner$id)
}

# A minimal stand-in for `ContextTorch`, holding what `CallbackSetRealMLP` accesses.
fake_realmlp_ctx = function(network, optimizer, global_step, total_epochs, n_batches) {
  ctx = new.env()
  ctx$network = network
  ctx$optimizer = optimizer
  ctx$global_step = global_step
  ctx$total_epochs = total_epochs
  ctx$loader_train = seq_len(n_batches)
  ctx
}

test_that("the schedules are correct", {
  expect_equal(realmlp_schedule("coslog4", 0), 0)
  # the four cycles end at t = 1/15, 3/15, 7/15 and 1
  expect_equal(realmlp_schedule("coslog4", c(1, 3, 7, 15) / 15), rep(0, 4))
  # and peak in between
  expect_equal(realmlp_schedule("coslog4", (sqrt(2) - 1) / 15), 1)
  expect_equal(realmlp_schedule("flat_cos", 0), 1)
  expect_equal(realmlp_schedule("flat_cos", 0.49), 1)
  expect_equal(realmlp_schedule("flat_cos", 0.75), 0.5)
  expect_equal(realmlp_schedule("flat_cos", 1), 0)
  expect_equal(realmlp_schedule("cos", 0.5), 0.5)
  expect_equal(realmlp_schedule("linear", 0.25), 0.75)
  expect_equal(realmlp_schedule("constant", 0.3), 1)
  expect_error(realmlp_schedule("foo", 0.3), "Unknown schedule")
})

test_that("realmlp_robust_scaling works", {
  x = cbind(
    a = c(1, 2, 3, 4, 100),
    # zero interquartile range: half the range is used
    b = c(0, 0, 0, 0, 4),
    # constant: scaled to zero
    c = rep(3, 5)
  )
  stats = realmlp_robust_scaling(x)
  expect_equal(stats$center, c(3, 0, 3))
  expect_equal(stats$scale, c(1 / 2, 1 / 2, 0))
})

test_that("nn_realmlp_plr works for all variants", {
  x = torch_randn(7, 3)
  for (cos_bias in c(TRUE, FALSE)) {
    for (densenet in c(TRUE, FALSE)) {
      for (activation in c("linear", "relu")) {
        plr = nn_realmlp_plr(3L, sigma = 0.1, d_hidden = 4L, d_out = 5L, activation = activation,
          densenet = densenet, cos_bias = cos_bias)
        out = plr(x)
        expect_equal(out$shape, c(7, 15))
        if (densenet) expect_equal(out[, 13:15], x)
        if (!densenet && activation == "relu") expect_true(as.logical((out >= 0)$all()))
      }
    }
  }
  expect_error(nn_realmlp_plr(3L, 0.1, d_hidden = 3L, d_out = 4L, activation = "linear",
    densenet = TRUE, cos_bias = FALSE), "must be even")
  expect_error(nn_realmlp_plr(3L, 0.1, d_hidden = 4L, d_out = 1L, activation = "linear",
    densenet = TRUE, cos_bias = TRUE), "at least 2")
})

test_that("nn_realmlp_plr embeds each feature with its own parameters", {
  plr = nn_realmlp_plr(2L, sigma = 1, d_hidden = 4L, d_out = 3L, activation = "linear",
    densenet = FALSE, cos_bias = TRUE)
  x = torch_randn(5, 2)
  out = plr(x)
  # the embedding of the second feature, computed directly
  h = torch_cos(2 * pi * x[, 2:2] * plr$weight_1[2, , ] + plr$bias_1[2, , ])
  expected = torch_matmul(h, plr$weight_2[2, , ]) + plr$bias_2[2, , ]
  expect_equal(as_array(out[, 4:6]), as_array(expected), tolerance = 1e-5)
})

test_that("nn_realmlp encodes the categorical features", {
  # levels: 1 and 2 levels are encoded in one column, 3 levels in three columns and 9 levels
  # (more than max_one_hot_cat_size - 1 = 8) are embedded
  cards = c(1L, 2L, 3L, 9L)
  x_cat = torch_tensor(cbind(
    c(1, 1, 1, 1, 1, 1),
    c(1, 2, 1, 2, 1, 2),
    c(1, 2, 3, 1, 2, 3),
    c(1, 2, 3, 4, 5, 9)
  ), dtype = torch_long())
  net = nn_realmlp(x_cat = x_cat, cat_cardinalities = cards, d_out = 3L, n_hidden_layers = 1L,
    hidden_width = 4L, embedding_size = 5L)
  expect_equal(net$onehot_cols, 1:3)
  expect_equal(net$emb_cols, 4L)
  one_hot = as_array(net$one_hot(x_cat))
  expect_equal(one_hot[, 1], rep(1, 6))
  expect_equal(one_hot[, 2], rep(c(1, -1), 3))
  expect_equal(one_hot[, 3:5], diag(3)[c(1:3, 1:3), ])
  # the input of the MLP: the one-hot encoding (1 + 1 + 3 columns) and the embedding
  expect_equal(net$blocks[[1L]]$d_in, 5L + 5L)
  expect_equal(net$embed(x_cat = x_cat)$shape, c(6, 10))
  # the embedding of the categorical feature with 9 levels
  expect_equal(as_array(net$embed(x_cat = x_cat)[, 6:10]), as_array(net$emb_weight$detach())[c(1:5, 9), ])
  # the constant one-hot column is scaled to zero
  expect_equal(as_array(net$embed(x_cat = x_cat)[, 1]), rep(0, 6))
  # purely categorical input arrives in the first argument
  expect_equal(net(x_cat)$shape, c(6, 3))

  # max_one_hot_cat_size counts the category for missing values
  net = nn_realmlp(x_cat = x_cat, cat_cardinalities = cards, d_out = 1L, n_hidden_layers = 0L,
    max_one_hot_cat_size = 3L)
  expect_equal(net$onehot_cols, 1:2)
  net = nn_realmlp(x_cat = x_cat, cat_cardinalities = cards, d_out = 1L, n_hidden_layers = 0L,
    max_one_hot_cat_size = 10L)
  expect_equal(net$onehot_cols, 1:4)
})

test_that("nn_realmlp preprocesses the numerical features", {
  x_num = torch_tensor(cbind(c(1, 2, 3, 4, 100), c(-1, 0, 1, 2, 3)))
  net = nn_realmlp(x_num = x_num, d_out = 1L, num_emb_type = "none", n_hidden_layers = 0L)
  expect_equal(net$blocks[[1L]]$d_in, 2L)
  scaled = cbind(c(-1, -0.5, 0, 0.5, 48.5), c(-1, -0.5, 0, 0.5, 1))
  expect_equal(as_array(net$embed(x_num)), scaled / sqrt(1 + scaled^2 / 9), tolerance = 1e-6)

  net = nn_realmlp(x_num = x_num, d_out = 1L, n_hidden_layers = 0L, plr_hidden_2 = 4L)
  expect_equal(net$blocks[[1L]]$d_in, 8L)
})

test_that("nn_realmlp has the architecture of upstream", {
  x_num = torch_randn(30, 3)
  net = nn_realmlp(x_num = x_num, d_out = 4L, n_hidden_layers = 3L, hidden_width = 16L)
  expect_length(net$blocks, 4L)
  blocks = map(seq_along(net$blocks), function(i) net$blocks[[i]])
  # the front scale is only part of the first layer
  expect_equal(map_lgl(blocks, function(b) !is.null(b$scale)), c(TRUE, FALSE, FALSE, FALSE))
  # the last layer has no activation and no dropout
  expect_equal(map_lgl(blocks, function(b) !is.null(b$act_fn)), c(TRUE, TRUE, TRUE, FALSE))
  expect_equal(map_lgl(blocks, function(b) !is.null(b$act_weight)), c(TRUE, TRUE, TRUE, FALSE))
  expect_equal(map_lgl(blocks, "has_dropout"), c(TRUE, TRUE, TRUE, FALSE))
  expect_equal(map_int(blocks, "d_in"), c(12L, 16L, 16L, 16L))
  expect_equal(net(x_num)$shape, c(30, 4))

  # without hidden layers, the configuration of the last layer takes precedence
  net = nn_realmlp(x_num = x_num, d_out = 4L, n_hidden_layers = 0L)
  expect_null(net$blocks[[1L]]$scale)

  net = nn_realmlp(x_num = x_num, d_out = 4L, add_front_scale = FALSE, use_parametric_act = FALSE)
  expect_null(net$blocks[[1L]]$scale)
  expect_null(net$blocks[[1L]]$act_weight)
})

test_that("the parametric activation interpolates between the identity and the activation", {
  block = nn_realmlp_block(3L, 4L, scale = FALSE, activation = "relu", parametric_act = TRUE)
  x = torch_randn(10, 4)
  with_no_grad(block$act_weight$fill_(0.25))
  expect_equal(as_array(block$activate(x)), as_array(x + 0.25 * (nnf_relu(x) - x)), tolerance = 1e-6)
})

test_that("the weights and biases are initialized from the data", {
  torch_manual_seed(1)
  x_num = torch_randn(200, 3) * 5 + 2
  net = nn_realmlp(x_num = x_num, d_out = 2L, n_hidden_layers = 2L, hidden_width = 32L)
  net$eval()
  x = net$embed(x_num)
  block = net$blocks[[1L]]
  pre = torch_matmul(x * block$scale, block$weight) * block$gain
  # the pre-activations (without bias) have unit standard deviation on the training data
  expect_equal(as_array(pre$std(dim = 1L, unbiased = FALSE)), rep(1, 32), tolerance = 1e-4)
  # the bias places the zero of each pre-activation inside the convex hull of the data
  pre = as_array(pre + block$bias)
  expect_true(all(apply(pre, 2L, min) <= 1e-6))
  expect_true(all(apply(pre, 2L, max) >= -1e-6))

  # the same holds for the following layers, whose input is the output of the previous one
  x = block(x)
  block = net$blocks[[2L]]
  pre = torch_matmul(x, block$weight) * block$gain
  expect_equal(as_array(pre$std(dim = 1L, unbiased = FALSE)), rep(1, 32), tolerance = 1e-4)

  # constant inputs do not lead to non-finite weights
  net = nn_realmlp(x_num = torch_ones(10, 2), d_out = 1L, num_emb_type = "none")
  expect_true(all(map_lgl(net$parameters, function(p) as.logical(p$isfinite()$all()))))
})

test_that("the memory estimates of the initialization are those of upstream", {
  # values of `get_n_forward()` of pytabkit's fitters for these configurations
  expect_n_forward = function(n_num, n_onehot, n_emb, embedding_size, num_emb_type, h1, h2, d_in,
    expected) {
    out = realmlp_init_n_forward(n_num, n_onehot, n_emb, embedding_size, num_emb_type, h1, h2, d_in)
    expect_equal(unlist(out), c(preprocessing = expected[1L], embedding = expected[2L]))
  }
  expect_n_forward(3L, 4L, 1L, 8L, "pbld", 16L, 4L, 24L, c(245L, 0L))
  expect_n_forward(5L, 0L, 0L, 8L, "pbld", 16L, 4L, 20L, c(311L, 0L))
  expect_n_forward(0L, 4L, 1L, 8L, "pbld", 16L, 4L, 12L, c(56L, 0L))
  expect_n_forward(4L, 1L, 2L, 8L, "pl", 8L, 5L, 37L, c(198L, 0L))
  expect_n_forward(4L, 5L, 0L, 8L, "pblrd", 16L, 4L, 21L, c(294L, 0L))
  expect_n_forward(4L, 5L, 0L, 8L, "plr", 16L, 4L, 21L, c(250L, 0L))
  expect_n_forward(4L, 5L, 1L, 5L, "none", 16L, 4L, 14L, c(9L, 20L))

  # 1 GB with 4 bytes per value
  expect_null(realmlp_init_subsample(1000L, 256L, 1))
  expect_null(realmlp_init_subsample(2^20, 256L, 1))
  expect_length(realmlp_init_subsample(2^20 + 1, 256L, 1), 2^20)
  expect_null(realmlp_init_subsample(10L, 0L, 1e-9))
})

test_that("the preprocessing and the initialization can use a subsample", {
  torch_manual_seed(1)
  x_num = torch_randn(100, 3)
  # preprocessing: 3 * (48 + 7) + 3 * 3 + 12 + 1 = 187 values, i.e. 4 * 187 * 10 bytes for 10 rows
  limit = 4 * 187 * 10 / 1024^3
  net = nn_realmlp(x_num = x_num, d_out = 1L, n_hidden_layers = 1L, hidden_width = 4L,
    init_ram_limit_gb = limit)
  center = as_array(net$num_scaler$center)[1L, ]
  # the statistics are computed on 10 of the 100 rows
  expect_false(isTRUE(all.equal(center, apply(as_array(x_num), 2L, stats::median))))
  expect_true(all(map_lgl(net$parameters, function(p) as.logical(p$isfinite()$all()))))
})

test_that("the initialization does not use dropout", {
  x_num = torch_randn(50, 3)
  torch_manual_seed(1)
  net1 = nn_realmlp(x_num = x_num, d_out = 2L, n_hidden_layers = 2L, hidden_width = 8L, p_drop = 0)
  torch_manual_seed(1)
  net2 = nn_realmlp(x_num = x_num, d_out = 2L, n_hidden_layers = 2L, hidden_width = 8L, p_drop = 0.5)
  expect_equal(map(net1$parameters, as_array), map(net2$parameters, as_array))
  expect_equal(net2$blocks[[1L]]$p_drop, 0.5)
  expect_true(net2$training)
})

test_that("binary classification returns the logit of the first class", {
  x_num = torch_randn(20, 3)
  net = nn_realmlp(x_num = x_num, d_out = 2L, binary = TRUE, n_hidden_layers = 1L,
    hidden_width = 4L)
  net$eval()
  out = net(x_num)
  expect_equal(out$shape, c(20, 1))
  logits = net$blocks[[2L]](net$blocks[[1L]](net$embed(x_num)))
  expect_equal(as_array(out[, 1]), as_array(logits[, 1] - logits[, 2]), tolerance = 1e-6)
  expect_error(nn_realmlp(x_num = x_num, d_out = 3L, binary = TRUE), "must have d_out = 2")
})

test_that("regression outputs are rescaled and clamped", {
  x_num = torch_randn(20, 3)
  y = rnorm(20, mean = 100, sd = 10)
  net = nn_realmlp(x_num = x_num, d_out = 1L, y = y, normalize_output = TRUE,
    clamp_output = TRUE, n_hidden_layers = 1L, hidden_width = 4L)
  expect_equal(as.numeric(net$y_std), sqrt(mean((y - mean(y))^2)), tolerance = 1e-5)
  net$eval()
  out = as_array(net(torch_randn(100, 3) * 100))
  expect_true(all(out >= min(y) - 1e-4 & out <= max(y) + 1e-4))
  # the predictions are on the scale of the target
  expect_true(abs(mean(as_array(net(x_num))) - mean(y)) < 3 * sd(y))
  # no clamping in training mode
  net$train()
  net$set_dropout(0)
  x = net$blocks[[2L]](net$blocks[[1L]](net$embed(x_num)))
  expect_equal(as_array(net(x_num)), as_array(x * net$y_std + net$y_mean), tolerance = 1e-5)
})

test_that("nn_realmlp gives informative errors", {
  expect_error(nn_realmlp(d_out = 1L), "at least one numerical or one categorical")
  expect_error(nn_realmlp(x_cat = torch_ones(3, 2, dtype = torch_long()), cat_cardinalities = 2L,
    d_out = 1L), "2 columns, but 1 cardinalities")
})

test_that("the parameter groups contain all parameters with the correct factors", {
  x_num = torch_randn(20, 3)
  x_cat = torch_tensor(cbind(c(rep(1, 10), rep(2, 10)), rep(1:10, 2)), dtype = torch_long())
  net = nn_realmlp(x_num = x_num, x_cat = x_cat, cat_cardinalities = c(2L, 10L), d_out = 2L,
    n_hidden_layers = 2L, hidden_width = 4L,
    lr_factors = list(plr = 0.2, scale = 5, bias = 0.3, act = 0.4))
  groups = net$param_groups()
  factors = net$param_group_factors()
  expect_length(groups, length(factors))
  expect_equal(sum(map_int(groups, function(g) length(g$params))), length(net$parameters))
  factor_of = function(param) {
    for (i in seq_along(groups)) {
      if (some(groups[[i]]$params, function(p) identical(p, param))) return(factors[[i]])
    }
    stop("parameter not found")
  }
  expect_equal(factor_of(net$plr$weight_1), list(lr_factor = 0.2, wd_factor = 1))
  expect_equal(factor_of(net$emb_weight), list(lr_factor = 1, wd_factor = 1))
  expect_equal(factor_of(net$blocks[[1L]]$scale), list(lr_factor = 5, wd_factor = 1))
  expect_equal(factor_of(net$blocks[[2L]]$weight), list(lr_factor = 1, wd_factor = 1))
  expect_equal(factor_of(net$blocks[[3L]]$bias), list(lr_factor = 0.3, wd_factor = 0))
  expect_equal(factor_of(net$blocks[[2L]]$act_weight), list(lr_factor = 0.4, wd_factor = 1))
  # the factors are unique per group
  expect_equal(anyDuplicated(factors), 0L)
})

test_that("CallbackSetRealMLP schedules the learning rate, the dropout and the weight decay", {
  torch_manual_seed(1)
  x_num = torch_randn(20, 3)
  net = nn_realmlp(x_num = x_num, d_out = 2L, n_hidden_layers = 1L, hidden_width = 4L,
    lr_factors = list(plr = 0.2, scale = 5, bias = 0.3, act = 0.4))
  opt = optim_adam(net$param_groups(), lr = 0.1)
  cb = CallbackSetRealMLP$new(lr_sched = "coslog4", wd = 0.5, wd_sched = "flat_cos",
    p_drop = 0.2, p_drop_sched = "flat_cos")
  # step 31 of 40 steps, i.e. t = 0.75
  cb$ctx = fake_realmlp_ctx(net, opt, global_step = 31L, total_epochs = 4L, n_batches = 10L)
  cb$on_begin()
  expect_equal(cb$state_dict(), list(base_lr = rep(0.1, length(opt$param_groups))))

  cb$on_batch_begin()
  expect_equal(cb$t, 0.75)
  lr_value = realmlp_schedule("coslog4", 0.75)
  factors = net$param_group_factors()
  expect_equal(map_dbl(opt$param_groups, "lr"), 0.1 * map_dbl(factors, "lr_factor") * lr_value)
  expect_equal(net$blocks[[1L]]$p_drop, 0.2 * 0.5)

  before = map(net$parameters, function(p) p$clone())
  cb$on_after_backward()
  wd = 0.5 * 0.5
  for (i in seq_along(opt$param_groups)) {
    decay = wd * 0.1 * lr_value * (factors[[i]]$lr_factor * factors[[i]]$wd_factor)^2
    for (param in opt$param_groups[[i]]$params) {
      name = names(net$parameters)[map_lgl(net$parameters, function(p) identical(p, param))]
      expect_equal(as_array(param), as_array(before[[name]] * (1 - decay)), tolerance = 1e-6)
    }
  }
  # the biases are not decayed
  expect_equal(as_array(net$blocks[[1L]]$bias), as_array(before[["blocks.0.bias"]]))
})

test_that("CallbackSetRealMLP keeps the base learning rate when resuming", {
  x_num = torch_randn(20, 3)
  net = nn_realmlp(x_num = x_num, d_out = 2L, n_hidden_layers = 1L, hidden_width = 4L)
  opt = optim_adam(net$param_groups(), lr = 0.1)
  # the learning rates of the optimizer are the scheduled ones
  for (i in seq_along(opt$param_groups)) opt$param_groups[[i]]$lr = 0.01
  cb = CallbackSetRealMLP$new(lr_sched = "constant", wd = 0, wd_sched = "constant",
    p_drop = 0, p_drop_sched = "constant")
  cb$ctx = fake_realmlp_ctx(net, opt, global_step = 2L, total_epochs = 2L, n_batches = 2L)
  cb$load_state_dict(list(base_lr = rep(0.1, length(opt$param_groups))))
  cb$on_begin()
  cb$on_batch_begin()
  expect_equal(map_dbl(opt$param_groups, "lr"),
    0.1 * map_dbl(net$param_group_factors(), "lr_factor"))
})

test_that("the learning rate follows the schedule during training", {
  task = tsk("iris")
  lrs = list()
  record = torch_callback("record", on_after_backward = function() {
    lrs[[length(lrs) + 1L]] <<- map_dbl(self$ctx$optimizer$param_groups, "lr")
  })
  learner = make_realmlp(epochs = 3L, batch_size = 50L, callbacks = record, opt.lr = 0.5)
  learner$train(task)
  lrs = do.call(rbind, lrs)
  # 3 batches per epoch
  expect_equal(nrow(lrs), 9L)
  factors = map_dbl(learner$model$network$param_group_factors(), "lr_factor")
  expected = t(sapply(realmlp_schedule("coslog4", (0:8) / 9), function(s) 0.5 * factors * s))
  expect_equal(lrs, expected)
  expect_equal(learner$model$callbacks$realmlp$base_lr, rep(0.5, length(factors)))
})

test_that("training continues the schedule when resuming from a checkpoint", {
  path = tempfile()
  task = tsk("iris")
  crash = torch_callback("crash",
    on_epoch_begin = function() if (self$ctx$epoch == 3L) stop("crash"))
  make = function(cbs) make_realmlp(epochs = 4L, batch_size = 50L, seed = 1L, callbacks = cbs)
  expect_error(make(list(t_clbk("checkpoint", freq = 1, path = path), crash))$train(task), "crash")

  seen = NULL
  record = function() {
    if (self$ctx$global_step == 8L) seen <<- map_dbl(self$ctx$optimizer$param_groups, "lr")
  }
  resumed = make(list(torch_callback("probe", on_after_backward = record)))
  resumed$param_set$set_values(resume = path)
  resumed$train(task)
  lr_resumed = seen
  expect_false(is.null(lr_resumed))

  seen = NULL
  make(list(torch_callback("probe", on_after_backward = record)))$train(task)
  expect_equal(lr_resumed, seen)
})

test_that("label smoothing is correct", {
  torch_manual_seed(1)
  input = torch_randn(6, 4)
  target = torch_tensor(c(1L, 2L, 3L, 4L, 1L, 2L), dtype = torch_long())
  loss = realmlp_ls_loss(nn_cross_entropy_loss(), eps = 0.1, binary = FALSE)
  soft = 0.9 * nnf_one_hot(target, 4L) + 0.1 / 4
  expected = -(soft * nnf_log_softmax(input, dim = 2L))$sum(dim = 2L)$mean()
  expect_equal(loss(input, target)$item(), expected$item(), tolerance = 1e-6)

  loss = realmlp_ls_loss(nn_cross_entropy_loss(reduction = "sum"), eps = 0.1, binary = FALSE,
    reduction = "sum")
  expect_equal(loss(input, target)$item(), 6 * expected$item(), tolerance = 1e-5)

  # binary: the same as label smoothing of two logits
  z = torch_randn(6, 1)
  y = torch_tensor(c(1, 0, 1, 1, 0, 0))$unsqueeze(2L)
  loss = realmlp_ls_loss(nn_bce_with_logits_loss(), eps = 0.1, binary = TRUE)
  logits = torch_cat(list(z, torch_zeros_like(z)), dim = 2L)
  soft = 0.9 * torch_cat(list(y, 1 - y), dim = 2L) + 0.05
  expected = -(soft * nnf_log_softmax(logits, dim = 2L))$sum(dim = 2L)$mean()
  expect_equal(loss(z, y)$item(), expected$item(), tolerance = 1e-6)
})

test_that("the learner wraps the configured loss", {
  task = tsk("iris")
  learner = make_realmlp()
  loss_fn = wrap_realmlp_loss(learner, task)
  expect_class(loss_fn, "realmlp_ls_loss")
  expect_equal(loss_fn$eps, 0.1)

  learner$param_set$set_values(ls_eps = 0)
  loss_fn = wrap_realmlp_loss(learner, task)
  expect_class(loss_fn, "nn_cross_entropy_loss")

  learner = make_realmlp(loss = t_loss("cross_entropy", reduction = "sum"))
  loss_fn = wrap_realmlp_loss(learner, task)
  expect_equal(loss_fn$reduction, "sum")

  # regression: the prediction and the target are standardized
  task = tsk("mtcars")
  learner = make_realmlp("regr")
  loss_fn = wrap_realmlp_loss(learner, task)
  expect_class(loss_fn, "realmlp_normalized_loss")
  y = task$truth()
  input = torch_tensor(y[1:5] + 1)$unsqueeze(2L)
  target = torch_tensor(y[1:5])$unsqueeze(2L)
  sd = sqrt(mean((y - mean(y))^2))
  expect_equal(loss_fn(input, target)$item(), 1 / sd^2, tolerance = 1e-5)

  learner$param_set$set_values(normalize_output = FALSE)
  loss_fn = wrap_realmlp_loss(learner, task)
  expect_class(loss_fn, "nn_mse_loss")
})

test_that("label smoothing requires the cross entropy loss", {
  loss = TorchLoss$new(nn_cross_entropy_loss, task_types = "classif", id = "custom")
  learner = make_realmlp(loss = loss)
  expect_error(learner$train(tsk("iris")), "requires the 'cross_entropy' loss")
  learner$param_set$set_values(ls_eps = 0)
  expect_error(learner$train(tsk("iris")), regexp = NA)
})

test_that("the defaults are those of RealMLP-TD", {
  learner = lrn("classif.realmlp")
  pv = learner$param_set$values
  expect_equal(pv$epochs, 256L)
  expect_equal(pv$batch_size, 256L)
  expect_equal(pv$batch_size_predict, 1024L)
  expect_true(pv$drop_last)
  expect_equal(pv$act, "selu")
  expect_equal(pv$ls_eps, 0.1)
  expect_equal(pv$opt.lr, 0.04)
  expect_equal(pv$opt.betas, c(0.9, 0.95))
  expect_equal(learner$optimizer$id, "adam")
  expect_equal(learner$loss$id, "cross_entropy")

  learner = lrn("regr.realmlp")
  pv = learner$param_set$values
  expect_equal(pv$act, "mish")
  expect_equal(pv$opt.lr, 0.2)
  expect_true(pv$normalize_output)
  expect_true(pv$clamp_output)
  expect_null(pv$ls_eps)
})

test_that("expect_learner_torch on multiclass, binary, mixed and regression tasks", {
  expect_learner_torch(make_realmlp(), task = tsk("iris"))
  expect_learner_torch(make_realmlp(), task = tsk("sonar"))
  expect_learner_torch(make_realmlp(predict_type = "prob"), task = tsk("german_credit"))
  expect_learner_torch(make_realmlp("regr"), task = tsk("mtcars"))
})

test_that("the learner works for all numerical embeddings", {
  task = tsk("german_credit")$filter(1:100)
  for (type in c("pbld", "pblrd", "pl", "plr", "none")) {
    learner = make_realmlp(num_emb_type = type)
    expect_error(learner$train(task), regexp = NA, info = type)
    expect_prediction(learner$predict(task))
  }
})

test_that("the learner works on tasks with only categorical features", {
  task = tsk("german_credit")$filter(1:100)
  task$select(task$feature_types[get("type") == "factor", get("id")])
  learner = make_realmlp(predict_type = "prob")
  learner$train(task)
  expect_prediction(learner$predict(task))
})

test_that("the learner encodes the columns in the order of the ingress tokens", {
  # after po("scale"), the order of task$feature_names differs from the order of the features
  # in task$feature_types
  task = po("scale")$train(list(tsk("german_credit")$filter(1:100)))[[1L]]
  learner = make_realmlp()
  learner$train(task)
  x_num = batchgetter_num(task$data(cols = ingress_num()$features(task)))
  x_cat = batchgetter_categ(task$data(cols = ingress_categ()$features(task)))
  network = learner$model$network
  expect_equal(as_array(network$num_scaler$center)[1L, ],
    apply(as_array(x_num), 2L, stats::median), tolerance = 1e-6)
  expect_equal(network$emb_cols, unname(which(categ_cardinalities(task) >= 9L)))
})

test_that("the hyperparameters affect the network", {
  task = tsk("iris")
  learner = make_realmlp(n_hidden_layers = 1L, hidden_width = 5L, act = "relu",
    use_parametric_act = FALSE, num_emb_type = "none", add_front_scale = FALSE)
  learner$train(task)
  network = learner$model$network
  expect_length(network$blocks, 2L)
  expect_equal(network$blocks[[1L]]$weight$shape, c(4, 5))
  expect_null(network$blocks[[1L]]$scale)
  expect_null(network$blocks[[1L]]$act_weight)
  expect_null(network$plr)
})

test_that("training with the defaults learns", {
  task = tsk("iris")
  learner = lrn("classif.realmlp", epochs = 30L, batch_size = 32L, seed = 1L)
  learner$train(task)
  expect_true(learner$predict(task)$score(msr("classif.ce")) < 0.1)

  task = tsk("mtcars")
  learner = lrn("regr.realmlp", epochs = 30L, batch_size = 8L, seed = 1L)
  learner$train(task)
  expect_true(learner$predict(task)$score(msr("regr.rsq")) > 0.7)
})

test_that("training is reproducible", {
  task = tsk("german_credit")$filter(1:100)
  learner = make_realmlp(seed = 1L, predict_type = "prob")
  p1 = learner$train(task)$predict(task)$prob
  p2 = learner$train(task)$predict(task)$prob
  expect_equal(p1, p2)
})

test_that("cloning also keeps parameter values", {
  learner = lrn("classif.realmlp", n_hidden_layers = 2L)
  learnerc = learner$clone(deep = TRUE)
  expect_deep_clone_mlr3torch(learner, learnerc)
  expect_equal(learnerc$param_set$values$n_hidden_layers, 2L)
  expect_equal(learnerc$param_set$values$opt.lr, 0.04)
})

test_that("informative errors for unsupported input", {
  learner = make_realmlp()
  expect_error(learner$train(tsk("lazy_iris")), "lazy_tensor")
})

test_that("learning rate schedulers cannot be combined with the learner", {
  learner = make_realmlp(callbacks = t_clbk("lr_step", step_size = 1))
  expect_error(learner$train(tsk("iris")), "cannot be combined with the learning rate scheduler callback\\(s\\) 'lr_step'")
  learner = make_realmlp(callbacks = t_clbk("lr_one_cycle", max_lr = 0.1))
  expect_error(learner$train(tsk("iris")), "'lr_one_cycle'")
  # other callbacks are fine
  learner = make_realmlp(callbacks = list(t_clbk("history"), t_clbk("progress")))
  expect_error(learner$train(tsk("iris")), regexp = NA)
})

test_that("frozen parameters are not decayed", {
  x_num = torch_randn(20, 3)
  net = nn_realmlp(x_num = x_num, d_out = 2L, n_hidden_layers = 1L, hidden_width = 4L)
  opt = optim_adam(net$param_groups(), lr = 0.1)
  cb = CallbackSetRealMLP$new(lr_sched = "constant", wd = 0.5, wd_sched = "constant",
    p_drop = 0, p_drop_sched = "constant")
  cb$ctx = fake_realmlp_ctx(net, opt, global_step = 1L, total_epochs = 1L, n_batches = 1L)
  net$blocks[[1L]]$weight$requires_grad_(FALSE)
  frozen = as_array(net$blocks[[1L]]$weight)
  decayed = as_array(net$blocks[[2L]]$weight)
  cb$on_begin()
  cb$on_batch_begin()
  cb$on_after_backward()
  expect_equal(as_array(net$blocks[[1L]]$weight), frozen)
  expect_true(all(abs(as_array(net$blocks[[2L]]$weight)) < abs(decayed)))
})

test_that("the parameters of the learner are required", {
  learner = make_realmlp()
  expect_true(all(c("act", "ls_eps", "p_drop", "num_emb_type") %in% learner$param_set$ids(tags = "required")))
  learner$param_set$values$act = NULL
  expect_error(learner$train(tsk("iris")), "Missing required parameters: act")
})
