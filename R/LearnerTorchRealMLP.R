# RealMLP (Holzmüller et al., NeurIPS 2024), ported from pytabkit 1.7.3
# (https://github.com/dholzmueller/pytabkit, commit c126ea51187c5080b91f28d352481dbd3b2194b0,
# Apache License 2.0). The defaults are those of `RealMLP_TD_CLASS` / `RealMLP_TD_REG` in
# `models/sklearn/default_params.py`.
#
# Deviations:
#  * Binary classification returns the difference of the two logits. this is mathematically equivalent
#    to predicting logit per class.
#  * Missing values are not supported.
#  * Weights are not rescaled when a pre-activation has zero standard deviation during the
#    initialization (upstream produces non-finite weights).

realmlp_schedules = c("coslog4", "flat_cos", "cos", "linear", "constant")

# `t` is the fraction of training steps already taken (`get_schedule()` in `training/scheduling.py`)
realmlp_schedule = function(name, t) {
  switch(name,
    constant = 1,
    coslog4 = 0.5 * (1 - cos(2 * pi * log2(1 + 15 * t))),
    flat_cos = if (t < 0.5) 1 else 0.5 * (1 + cos(pi * (t - 0.5) / 0.5)),
    cos = 0.5 * (1 + cos(pi * t)),
    linear = 1 - t,
    stopf("Unknown schedule '%s'.", name)
  )
}

realmlp_smooth_clip = function(x, max_abs_value = 3) {
  x / torch_sqrt(1 + x^2 / max_abs_value^2)
}

realmlp_activation = function(name) {
  switch(name,
    selu = function(x) nnf_selu(x),
    mish = function(x) x * torch_tanh(nnf_softplus(x)),
    relu = function(x) nnf_relu(x),
    silu = function(x) x * torch_sigmoid(x),
    gelu = function(x) nnf_gelu(x),
    elu = function(x) nnf_elu(x),
    stopf("Unknown activation '%s'.", name)
  )
}

# Median and inverse interquartile range of each column (`MedianCenterFactory`,
# `RobustScaleFactory`). Half the range is used when the IQR is zero, constant columns get scale 0.
realmlp_robust_scaling = function(x) {
  x = as.matrix(x)
  storage.mode(x) = "double"
  # type 7 matches numpy's quantile()
  q = matrix(apply(x, 2L, stats::quantile, probs = c(0.25, 0.5, 0.75), names = FALSE, type = 7L),
    nrow = 3L)
  quant_diff = q[3L, ] - q[1L, ]
  constant_iqr = quant_diff == 0
  if (any(constant_iqr)) {
    quant_diff[constant_iqr] = 0.5 * apply(x[, constant_iqr, drop = FALSE], 2L, function(col) max(col) - min(col))
  }
  scale = 1 / (quant_diff + 1e-30)
  scale[quant_diff == 0] = 0
  list(center = q[2L, ], scale = scale)
}

nn_realmlp_scaler = nn_module("nn_realmlp_scaler",
  initialize = function(center, scale) {
    self$center = nn_buffer(torch_tensor(center, dtype = torch_float())$unsqueeze(1L))
    self$scale = nn_buffer(torch_tensor(scale, dtype = torch_float())$unsqueeze(1L))
  },
  forward = function(input) {
    realmlp_smooth_clip((input - self$center) * self$scale)
  }
)

# Periodic embeddings (`PLREmbeddingsFactory`): (batch, n_features) -> (batch, n_features * d_out).
# With `densenet`, the input is appended as the last n_features columns.
nn_realmlp_plr = nn_module("nn_realmlp_plr",
  initialize = function(n_features, sigma, d_hidden, d_out, activation, densenet, cos_bias) {
    self$activation = assert_choice(activation, c("linear", "relu"))
    self$densenet = assert_flag(densenet)
    self$cos_bias = assert_flag(cos_bias)
    if (!cos_bias && d_hidden %% 2L != 0L) {
      stopf("The first hidden dimension of the periodic embeddings (plr_hidden_1) must be even for the sin/cos variant, but is %i.", d_hidden) # nolint
    }
    d_linear = d_out - as.integer(densenet)
    if (d_linear < 1L) {
      stopf("The output dimension of the periodic embeddings (plr_hidden_2) must be at least %i, but is %i.", 1L + as.integer(densenet), d_out) # nolint
    }
    if (cos_bias) {
      self$weight_1 = nn_parameter(sigma * torch_randn(n_features, 1L, d_hidden))
      self$bias_1 = nn_parameter(pi * (2 * torch_rand(n_features, 1L, d_hidden) - 1))
    } else {
      self$weight_1 = nn_parameter(sigma * torch_randn(n_features, 1L, d_hidden %/% 2L))
    }
    self$weight_2 = nn_parameter((2 * torch_rand(n_features, d_hidden, d_linear) - 1) / sqrt(d_hidden))
    self$bias_2 = nn_parameter((2 * torch_rand(n_features, 1L, d_linear) - 1) / sqrt(d_hidden))
  },
  forward = function(input) {
    # the features become a batch dimension of the matrix multiplications
    x = input$t()$unsqueeze(-1L)
    x = 2 * pi * torch_matmul(x, self$weight_1)
    x = if (self$cos_bias) {
      torch_cos(x + self$bias_1)
    } else {
      torch_cat(list(torch_cos(x), torch_sin(x)), dim = -1L)
    }
    x = torch_matmul(x, self$weight_2) + self$bias_2
    if (self$activation == "relu") x = nnf_relu(x)
    x = x$transpose(1L, 2L)$flatten(start_dim = 2L)
    if (self$densenet) x = torch_cat(list(x, input), dim = -1L)
    x
  }
)

# [scale] - weight - bias - [activation - dropout], with the NTK parametrization of the weight.
nn_realmlp_block = nn_module("nn_realmlp_block",
  initialize = function(d_in, d_out, scale, activation = NULL, parametric_act = FALSE, dropout = FALSE) {
    self$d_in = d_in
    self$gain = 1 / sqrt(d_in)
    self$scale = if (scale) nn_parameter(torch_ones(1L, d_in))
    self$weight = nn_parameter(torch_randn(d_in, d_out))
    self$bias = nn_parameter(torch_zeros(1L, d_out))
    self$act_fn = if (!is.null(activation)) realmlp_activation(activation)
    self$act_weight = if (!is.null(activation) && parametric_act) nn_parameter(torch_ones(1L, d_out))
    self$has_dropout = dropout
    self$p_drop = 0
  },
  forward = function(input) {
    x = if (is.null(self$scale)) input else input * self$scale
    x = torch_matmul(x, self$weight) * self$gain + self$bias
    self$activate(x)
  },
  activate = function(x) {
    if (!is.null(self$act_fn)) {
      x = if (is.null(self$act_weight)) self$act_fn(x) else x + (self$act_fn(x) - x) * self$act_weight
    }
    if (self$has_dropout && self$p_drop > 0) {
      x = nnf_dropout(x, p = self$p_drop, training = self$training)
    }
    x
  },
  # Upstream's `weight_init_mode = "std"` and `bias_init_mode = "he+5"`: each pre-activation gets
  # unit standard deviation on `input`, and is zero at a random convex combination of 5 rows of it.
  # Returns the output of the block.
  init_from_data = function(input) {
    x = if (is.null(self$scale)) input else input * self$scale
    weight = torch_randn(self$d_in, self$weight$shape[2L])
    std = torch_matmul(x, self$gain * weight)$std(dim = 1L, unbiased = FALSE, keepdim = TRUE)
    self$weight$copy_(weight / torch_where(std > 0, std, torch_ones_like(std)))

    out = torch_matmul(x, self$weight) * self$gain
    d_out = out$shape[2L]
    idx = torch_randint(1L, out$shape[1L] + 1L, size = c(d_out, 5L), dtype = torch_long())
    simplex_weights = torch_empty(d_out, 5L)$exponential_()
    simplex_weights = simplex_weights / simplex_weights$sum(dim = 2L, keepdim = TRUE)
    selected = torch_gather(out$t(), dim = 2L, index = idx)
    self$bias$copy_(-(selected * simplex_weights)$sum(dim = 2L)$unsqueeze(1L))

    self$activate(out + self$bias)
  }
)

# Upstream subsamples the data of an initialization step if it would need more than 1 GB, based on
# the step's estimate of values per observation (`get_n_forward()` of the fitters). This returns
# these estimates for the steps before the MLP: the step fitting the preprocessing statistics
# (`ConcatParallelFitter` with numerical embeddings, otherwise the robust scaling) and, without
# numerical embeddings, the categorical embedding.
realmlp_init_n_forward = function(n_num, n_onehot, n_emb, embedding_size, num_emb_type,
  plr_hidden_1, plr_hidden_2, d_in) {
  emb = if (n_emb > 0L) n_emb + 2L * n_emb * embedding_size else 0L
  if (num_emb_type == "none") {
    return(list(
      preprocessing = n_num + n_onehot,
      embedding = if (n_emb > 0L) n_num + n_onehot + emb else 0L
    ))
  }
  cos_bias = num_emb_type %in% c("pbld", "pblrd")
  hidden_2 = if (cos_bias) 2L * plr_hidden_2 - 1L else plr_hidden_2
  if (num_emb_type %in% c("pblrd", "plr")) hidden_2 = hidden_2 + plr_hidden_2
  plr = n_num * ((if (cos_bias) 3L * plr_hidden_1 else as.integer(2.5 * plr_hidden_1)) + hidden_2)
  # the renaming step only counts when there are categorical features
  has_cat = n_onehot + n_emb > 0L
  numerical = if (n_num > 0L) (if (has_cat) 4L else 3L) * n_num + plr else 0L
  categorical = (if (n_emb > 0L) n_onehot + 2L * n_emb else 0L) + 4L * n_onehot +
    (if (n_emb > 0L) n_onehot + emb else 0L)
  # + 1 for the target
  list(preprocessing = numerical + categorical + d_in + 1L, embedding = 0L)
}

# Indices of a random subsample of `n` rows if a step needing `n_forward` values per row (4 bytes
# each) exceeds `ram_limit_gb`, `NULL` otherwise (`Fitter._fit_transform_subsample()`).
realmlp_init_subsample = function(n, n_forward, ram_limit_gb) {
  if (n_forward <= 0) {
    return(NULL)
  }
  max_n = max(1L, floor(ram_limit_gb * 1024^3 / (4 * n_forward)))
  if (n <= max_n) {
    return(NULL)
  }
  as_array(torch_randperm(n))[seq_len(max_n)] + 1L
}

realmlp_subsample_rows = function(x, n_forward, ram_limit_gb) {
  sel = realmlp_init_subsample(x$shape[1L], n_forward, ram_limit_gb)
  if (is.null(sel)) x else x[sel, , drop = FALSE]
}

# The RealMLP network including its preprocessing. The preprocessing statistics and the
# data-dependent initialization are computed from the training data `x_num` / `x_cat` (1-based
# codes). For binary classification (`d_out = 2`), the difference of the two logits is returned.
nn_realmlp = nn_module("nn_realmlp",
  initialize = function(x_num = NULL, x_cat = NULL, cat_cardinalities = integer(0), d_out,
    binary = FALSE, y = NULL, normalize_output = FALSE, clamp_output = FALSE,
    n_hidden_layers = 3L, hidden_width = 256L, act = "selu", use_parametric_act = TRUE,
    p_drop = 0.15, add_front_scale = TRUE, num_emb_type = "pbld", plr_sigma = 0.1,
    plr_hidden_1 = 16L, plr_hidden_2 = 4L, max_one_hot_cat_size = 9L, embedding_size = 8L,
    lr_factors = list(plr = 0.1, scale = 6, bias = 0.1, act = 0.1), classification = FALSE,
    init_ram_limit_gb = 1) {
    assert_class(x_num, "torch_tensor", null.ok = TRUE)
    assert_class(x_cat, "torch_tensor", null.ok = TRUE)
    cat_cardinalities = assert_integerish(cat_cardinalities, lower = 1L, any.missing = FALSE,
      coerce = TRUE)
    d_out = assert_int(d_out, lower = 1L, coerce = TRUE)
    assert_flag(binary)
    if (binary && d_out != 2L) {
      stopf("A binary classification network must have d_out = 2, but has %i.", d_out)
    }
    n_hidden_layers = assert_int(n_hidden_layers, lower = 0L, coerce = TRUE)
    hidden_width = assert_int(hidden_width, lower = 1L, coerce = TRUE)
    assert_number(p_drop, lower = 0, upper = 1)
    assert_choice(num_emb_type, c("pbld", "pblrd", "pl", "plr", "none"))
    max_one_hot_cat_size = assert_int(max_one_hot_cat_size, coerce = TRUE)
    embedding_size = assert_int(embedding_size, lower = 1L, coerce = TRUE)
    assert_flag(classification)
    assert_number(init_ram_limit_gb, lower = 0)
    assert_list(lr_factors, types = "numeric")
    assert_permutation(names(lr_factors), c("plr", "scale", "bias", "act"))

    n_num = if (is.null(x_num)) 0L else x_num$shape[2L]
    n_cat = if (is.null(x_cat)) 0L else x_cat$shape[2L]
    if (n_num + n_cat == 0L) {
      stopf("nn_realmlp() requires at least one numerical or one categorical feature.")
    }
    if (n_cat != length(cat_cardinalities)) {
      stopf("x_cat has %i columns, but %i cardinalities were given.", n_cat, length(cat_cardinalities))
    }
    n = if (n_num > 0L) x_num$shape[1L] else x_cat$shape[1L]
    self$n_num = n_num
    self$binary = binary
    self$plr = NULL
    self$lr_factors = lr_factors

    d_in = 0L
    if (n_num > 0L) {
      self$plr = if (num_emb_type != "none") {
        nn_realmlp_plr(n_num, sigma = plr_sigma, d_hidden = plr_hidden_1, d_out = plr_hidden_2,
          activation = if (num_emb_type %in% c("pblrd", "plr")) "relu" else "linear",
          densenet = num_emb_type %in% c("pbld", "pblrd"),
          cos_bias = num_emb_type %in% c("pbld", "pblrd"))
      }
      d_in = d_in + if (is.null(self$plr)) n_num else n_num * plr_hidden_2
    }

    # like upstream, `max_one_hot_cat_size` counts a category for missing values
    one_hot = cat_cardinalities + 1L <= max_one_hot_cat_size
    self$onehot_cols = which(one_hot)
    self$emb_cols = which(!one_hot)
    if (length(self$onehot_cols)) {
      cards = cat_cardinalities[one_hot]
      # one or two levels are encoded in a single column (1 / -1 for two levels)
      widths = ifelse(cards <= 2L, 1L, cards)
      # the column and value that each level (row offset of its feature + level) sets
      row_offsets = c(0L, cumsum(cards))[seq_along(cards)]
      col_offsets = c(0L, cumsum(widths))[seq_along(cards)]
      columns = unlist(map(seq_along(cards), function(j) {
        col_offsets[j] + if (cards[j] <= 2L) rep(1L, cards[j]) else seq_len(cards[j])
      }))
      values = unlist(map(seq_along(cards), function(j) if (cards[j] == 2L) c(1, -1) else rep(1, cards[j])))
      self$onehot_width = sum(widths)
      self$onehot_columns = nn_buffer(torch_tensor(matrix(columns, ncol = 1L), dtype = torch_long()))
      self$onehot_values = nn_buffer(torch_tensor(matrix(values, ncol = 1L), dtype = torch_float()))
      self$onehot_offsets = nn_buffer(torch_tensor(matrix(row_offsets, nrow = 1L), dtype = torch_long()))
      d_in = d_in + sum(widths)
    }
    if (length(self$emb_cols)) {
      cards = cat_cardinalities[!one_hot]
      self$emb_offsets = nn_buffer(torch_tensor(matrix(c(0L, cumsum(cards))[seq_along(cards)], nrow = 1L),
        dtype = torch_long()))
      self$emb_weight = nn_parameter(torch_randn(sum(cards), embedding_size))
      d_in = d_in + length(cards) * embedding_size
    }

    sizes = c(rep(hidden_width, n_hidden_layers), d_out)
    blocks = vector("list", length(sizes))
    for (i in seq_along(sizes)) {
      is_last = i == length(sizes)
      blocks[[i]] = nn_realmlp_block(
        d_in = if (i == 1L) d_in else sizes[i - 1L],
        d_out = sizes[i],
        # upstream's config of the last layer takes precedence over the one of the first layer
        scale = i == 1L && !is_last && add_front_scale,
        activation = if (!is_last) act,
        parametric_act = use_parametric_act,
        dropout = !is_last
      )
    }
    self$blocks = nn_module_list(blocks)

    self$normalize_output = assert_flag(normalize_output)
    self$clamp_output = assert_flag(clamp_output)
    if (normalize_output || clamp_output) {
      assert_numeric(y, any.missing = FALSE, len = n)
    }
    if (normalize_output) {
      self$y_mean = nn_buffer(torch_tensor(mean(y), dtype = torch_float()))
      self$y_std = nn_buffer(torch_tensor(sqrt(mean((y - mean(y))^2)), dtype = torch_float()))
    }
    if (clamp_output) {
      self$y_min = nn_buffer(torch_tensor(min(y), dtype = torch_float()))
      self$y_max = nn_buffer(torch_tensor(max(y), dtype = torch_float()))
    }

    # preprocessing statistics and initialization, subsampled like upstream
    n_forward = realmlp_init_n_forward(n_num = n_num, n_onehot = self$onehot_width %??% 0L,
      n_emb = length(self$emb_cols), embedding_size = embedding_size, num_emb_type = num_emb_type,
      plr_hidden_1 = plr_hidden_1, plr_hidden_2 = plr_hidden_2, d_in = d_in)
    subsample = function(rows, n_values) {
      sel = realmlp_init_subsample(length(rows), n_values, init_ram_limit_gb)
      if (is.null(sel)) rows else rows[sel]
    }
    rows = subsample(seq_len(n), n_forward$preprocessing)
    if (n_num > 0L) {
      stats = realmlp_robust_scaling(as_array(x_num[rows, , drop = FALSE]))
      self$num_scaler = nn_realmlp_scaler(stats$center, stats$scale)
    }
    if (length(self$onehot_cols)) {
      stats = realmlp_robust_scaling(as_array(self$one_hot(x_cat[rows, , drop = FALSE])))
      self$cat_scaler = nn_realmlp_scaler(stats$center, stats$scale)
    }
    rows = subsample(rows, n_forward$embedding)
    if (classification) {
      # soft labels and label smoothing
      rows = subsample(subsample(rows, d_out), d_out)
    }
    with_no_grad({
      x = self$embed(
        if (n_num > 0L) x_num[rows, , drop = FALSE],
        if (n_cat > 0L) x_cat[rows, , drop = FALSE]
      )
      # upstream initializes in evaluation mode, i.e. without dropout
      self$eval()
      for (i in seq_along(self$blocks)) {
        block = self$blocks[[i]]
        if (!is.null(block$scale)) x = realmlp_subsample_rows(x, block$d_in, init_ram_limit_gb)
        x = block$init_from_data(x)
        if (!is.null(block$act_fn)) x = realmlp_subsample_rows(x, x$shape[2L], init_ram_limit_gb)
      }
    })
    self$set_dropout(p_drop)
    self$train()
  },
  one_hot = function(x_cat) {
    idx = x_cat[, self$onehot_cols, drop = FALSE] + self$onehot_offsets
    columns = nnf_embedding(idx, self$onehot_columns)$squeeze(3L)
    values = nnf_embedding(idx, self$onehot_values)$squeeze(3L)
    torch_zeros(idx$shape[1L], self$onehot_width, device = values$device)$scatter(2L, columns, values)
  },
  embed = function(x_num = NULL, x_cat = NULL) {
    parts = list()
    if (!is.null(x_num)) {
      x = self$num_scaler(x_num)
      parts[[length(parts) + 1L]] = if (is.null(self$plr)) x else self$plr(x)
    }
    if (length(self$onehot_cols)) {
      parts[[length(parts) + 1L]] = self$cat_scaler(self$one_hot(x_cat))
    }
    if (length(self$emb_cols)) {
      idx = x_cat[, self$emb_cols, drop = FALSE] + self$emb_offsets
      parts[[length(parts) + 1L]] = nnf_embedding(idx, self$emb_weight)$flatten(start_dim = 2L)
    }
    if (length(parts) == 1L) parts[[1L]] else torch_cat(parts, dim = 2L)
  },
  set_dropout = function(p) {
    for (i in seq_along(self$blocks)) {
      # `self$blocks[[i]]$p_drop = p` would call `[[<-` on the module list
      block = self$blocks[[i]]
      block$p_drop = p
    }
    invisible(self)
  },
  # The parameters grouped by their learning rate and weight decay factors, which
  # `$param_group_factors()` returns in the same order.
  param_groups = function() {
    map(private$group_entries(), function(group) list(params = map(group, "param")))
  },
  param_group_factors = function() {
    map(private$group_entries(), function(group) group[[1L]][c("lr_factor", "wd_factor")])
  },
  forward = function(x_num = NULL, x_cat = NULL) {
    # a network with a single input is called by position, so categorical-only input arrives here
    if (self$n_num == 0L && is.null(x_cat)) {
      x_cat = x_num
      x_num = NULL
    }
    x = self$embed(x_num, x_cat)
    for (i in seq_along(self$blocks)) {
      x = self$blocks[[i]](x)
    }
    if (self$binary) x = x[, 1:1] - x[, 2:2]
    if (self$normalize_output) x = x * self$y_std + self$y_mean
    if (self$clamp_output && !self$training) {
      x = torch_minimum(torch_maximum(x, self$y_min), self$y_max)
    }
    x
  },
  private = list(
    group_entries = function() {
      lr_factors = self$lr_factors
      entries = list()
      add = function(param, lr_factor, wd_factor) {
        entries[[length(entries) + 1L]] <<- list(param = param, lr_factor = lr_factor, wd_factor = wd_factor)
      }
      if (!is.null(self$plr)) {
        walk(self$plr$parameters, add, lr_factor = lr_factors$plr, wd_factor = 1)
      }
      if (!is.null(self$emb_weight)) add(self$emb_weight, 1, 1)
      for (i in seq_along(self$blocks)) {
        block = self$blocks[[i]]
        if (!is.null(block$scale)) add(block$scale, lr_factors$scale, 1)
        add(block$weight, 1, 1)
        add(block$bias, lr_factors$bias, 0)
        if (!is.null(block$act_weight)) add(block$act_weight, lr_factors$act, 1)
      }
      if (length(entries) != length(self$parameters)) {
        stopf("Internal error: not all parameters of nn_realmlp are assigned to a parameter group.")
      }
      keys = map_chr(entries, function(e) paste(e$lr_factor, e$wd_factor))
      unname(split(entries, factor(keys, levels = unique(keys))))
    }
  )
)

# Label smoothing towards the uniform distribution. The cross entropy is linear in the target, so
# this is `(1 - eps) * loss(true target) + eps * loss(uniform target)`.
realmlp_ls_loss = nn_module("realmlp_ls_loss",
  initialize = function(loss, eps, binary, reduction = "mean") {
    self$loss = loss
    self$eps = assert_number(eps, lower = 0, upper = 1)
    self$binary = assert_flag(binary)
    self$reduction = assert_choice(reduction, c("mean", "sum"))
  },
  forward = function(input, target) {
    if (self$binary) {
      return(self$loss(input, (1 - self$eps) * target + self$eps / 2))
    }
    uniform = -nnf_log_softmax(input, dim = 2L)$mean(dim = 2L)
    uniform = if (self$reduction == "mean") uniform$mean() else uniform$sum()
    (1 - self$eps) * self$loss(input, target) + self$eps * uniform
  }
)

# Standardizes prediction and target with the training target's mean and standard deviation.
realmlp_normalized_loss = nn_module("realmlp_normalized_loss",
  initialize = function(loss, mean, std) {
    self$loss = loss
    self$mean = mean
    self$std = std + 1e-30
  },
  forward = function(input, target) {
    self$loss((input - self$mean) / self$std, (target - self$mean) / self$std)
  }
)

# The learning rate, dropout and weight decay schedules. The optimizer must have been created from
# `network$param_groups()`. The learning rate of group i is `base_lr[i] * lr_factor[i] * lr_sched(t)`
# and the weight decay is applied after the backward pass.
CallbackSetRealMLP = R6Class("CallbackSetRealMLP",
  inherit = CallbackSet,
  lock_objects = FALSE,
  public = list(
    initialize = function(lr_sched, wd, wd_sched, p_drop, p_drop_sched) {
      self$lr_sched = assert_choice(lr_sched, realmlp_schedules)
      self$wd = assert_number(wd, lower = 0)
      self$wd_sched = assert_choice(wd_sched, realmlp_schedules)
      self$p_drop = assert_number(p_drop, lower = 0, upper = 1)
      self$p_drop_sched = assert_choice(p_drop_sched, realmlp_schedules)
    },
    on_begin = function() {
      groups = self$ctx$optimizer$param_groups
      private$.factors = self$ctx$network$param_group_factors()
      if (length(private$.factors) != length(groups)) {
        stopf("Internal error: the optimizer has %i parameter groups, but the network defines %i.",
          length(groups), length(private$.factors))
      }
      # when resuming, the base learning rates come from `$load_state_dict()`
      if (is.null(private$.base_lr)) {
        private$.base_lr = map_dbl(groups, "lr")
      }
    },
    on_batch_begin = function() {
      self$t = (self$ctx$global_step - 1) / (self$ctx$total_epochs * length(self$ctx$loader_train))
      lr_value = realmlp_schedule(self$lr_sched, self$t)
      for (i in seq_along(private$.factors)) {
        self$ctx$optimizer$param_groups[[i]]$lr = private$.base_lr[i] * private$.factors[[i]]$lr_factor * lr_value
      }
      self$ctx$network$set_dropout(self$p_drop * realmlp_schedule(self$p_drop_sched, self$t))
    },
    on_after_backward = function() {
      wd = self$wd * realmlp_schedule(self$wd_sched, self$t)
      if (wd == 0) {
        return(invisible(NULL))
      }
      with_no_grad({
        for (i in seq_along(private$.factors)) {
          factors = private$.factors[[i]]
          # the group's learning rate already contains the learning rate factor once
          decay = wd * self$ctx$optimizer$param_groups[[i]]$lr * factors$lr_factor * factors$wd_factor^2
          if (decay != 0) {
            for (param in self$ctx$optimizer$param_groups[[i]]$params) {
              if (param$requires_grad) param$mul_(1 - decay)
            }
          }
        }
      })
    },
    state_dict = function() {
      list(base_lr = private$.base_lr)
    },
    load_state_dict = function(state_dict) {
      private$.base_lr = state_dict$base_lr
      invisible(NULL)
    }
  ),
  private = list(
    .base_lr = NULL,
    .factors = NULL
  )
)

# Wraps the configured loss with label smoothing (classification) or target standardization
# (regression).
realmlp_wrap_loss = function(loss_fn, task, loss, param_vals, learner_id) {
  if (task$task_type == "classif" && param_vals$ls_eps > 0) {
    if (loss$id != "cross_entropy") {
      stopf("Learner '%s': label smoothing (ls_eps > 0) requires the 'cross_entropy' loss, but the loss is '%s'. Set ls_eps = 0 to use a different loss.", learner_id, loss$id) # nolint
    }
    loss_fn = realmlp_ls_loss(loss_fn, eps = param_vals$ls_eps,
      binary = "twoclass" %in% task$properties,
      reduction = loss$param_set$values$reduction %??% "mean")
  }
  if (task$task_type == "regr" && param_vals$normalize_output) {
    y = task$truth()
    loss_fn = realmlp_normalized_loss(loss_fn, mean = mean(y), std = sqrt(mean((y - mean(y))^2)))
  }
  loss_fn
}

#' @title RealMLP
#'
#' @templateVar name realmlp
#' @templateVar task_types classif, regr
#' @templateVar param_vals n_hidden_layers = 2, hidden_width = 32
#' @template params_learner
#' @template learner
#' @template learner_example
#'
#' @description
#' RealMLP is an MLP for tabular data with tuned defaults (RealMLP-TD), introduced by
#' `r cite_bib("holzmueller2024better")`.
#' The learner implements the preprocessing, architecture and training recipe of the reference
#' implementation.
#'
#' It is a special learner, as it implements it's own learning rate scheduling per parameter group,
#' as well as custom weight decay handling.
#' Configuring a custom learning reate scheduler as a callback or setting the weight decay of
#' an optimizer will therefore lead to unexpected results.
#' @section Parameters:
#' Changed defaults of [`LearnerTorch`]:
#' * `epochs = 256`
#' * `batch_size = 256`
#' * `batch_size_predict = 1024`
#' * `drop_last = TRUE`
#' and the optimizer is *adam* with
#' * `betas = c(0.9, 0.95)`
#' * `lr = 0.04` (classification) or `lr = 0.2` (regression).
#'
#' Parameters from [`LearnerTorch`], as well as:
#' * `n_hidden_layers` :: `integer(1)`\cr
#'   The number of hidden layers. Default is `3`.
#' * `hidden_width` :: `integer(1)`\cr
#'   The width of the hidden layers. Default is `256`.
#' * `act` :: `character(1)`\cr
#'   The activation function, one of `"selu"`, `"mish"`, `"relu"`, `"silu"`, `"gelu"` and
#'   `"elu"`. Default is `"selu"` for classification and `"mish"` for regression.
#' * `use_parametric_act` :: `logical(1)`\cr
#'   Whether to use parametric activations \eqn{x + \alpha (f(x) - x)} with a learned \eqn{\alpha}
#'   per neuron, initialized with one. Default is `TRUE`.
#' * `p_drop` :: `numeric(1)`\cr
#'   The dropout probability. Default is `0.15`.
#' * `p_drop_sched` :: `character(1)`\cr
#'   The schedule of the dropout probability. Default is `"flat_cos"`.
#' * `wd` :: `numeric(1)`\cr
#'   The decoupled weight decay. Biases are not decayed. Default is `0.02`.
#' * `wd_sched` :: `character(1)`\cr
#'   The schedule of the weight decay. Default is `"flat_cos"`.
#' * `lr_sched` :: `character(1)`\cr
#'   The schedule of the learning rate. Default is `"coslog4"`.
#' * `bias_lr_factor`, `act_lr_factor`, `scale_lr_factor`, `plr_lr_factor` :: `numeric(1)`\cr
#'   The learning rate factors of the biases (default `0.1`), the parametric activations
#'   (default `0.1`), the scaling layer (default `6`) and the periodic embeddings (default `0.1`).
#'   All other parameters have the factor `1`.
#' * `add_front_scale` :: `logical(1)`\cr
#'   Whether the first layer starts with a learned elementwise scaling. Default is `TRUE`.
#' * `num_emb_type` :: `character(1)`\cr
#'   The embedding of the numerical features, one of:
#'   * `"pbld"` (default): \eqn{\mathrm{concat}(W_2 \cos(2 \pi w_1 x + b_1) + b_2, x)}.
#'   * `"pblrd"`: as `"pbld"`, with a ReLU after the second linear layer.
#'   * `"pl"`: \eqn{W_2 \mathrm{concat}(\cos(2 \pi w_1 x), \sin(2 \pi w_1 x)) + b_2}.
#'   * `"plr"`: as `"pl"`, with a ReLU after the second linear layer.
#'   * `"none"`: no embedding.
#' * `plr_sigma` :: `numeric(1)`\cr
#'   The standard deviation of the initialization of the frequencies \eqn{w_1}. Default is `0.1`.
#' * `plr_hidden_1` :: `integer(1)`\cr
#'   The number of frequencies (`"pbld"`, `"pblrd"`) or twice the number of frequencies
#'   (`"pl"`, `"plr"`, must be even) per feature. Default is `16`.
#' * `plr_hidden_2` :: `integer(1)`\cr
#'   The embedding size per feature, including the appended input of `"pbld"` and `"pblrd"`.
#'   Default is `4`.
#' * `max_one_hot_cat_size` :: `integer(1)`\cr
#'   Categorical features with at most `max_one_hot_cat_size - 1` levels are one-hot encoded, all
#'   others are embedded. Default is `9`.
#' * `embedding_size` :: `integer(1)`\cr
#'   The embedding size of the categorical features. Default is `8`.
#'
#' Only for classification:
#' * `ls_eps` :: `numeric(1)`\cr
#'   The label smoothing parameter. Requires the `cross_entropy` loss. The smoothing term ignores
#'   the loss's `class_weight` and `ignore_index`. Default is `0.1`.
#'
#' Only for regression:
#' * `normalize_output` :: `logical(1)`\cr
#'   Whether to standardize the target during training. Default is `TRUE`.
#' * `clamp_output` :: `logical(1)`\cr
#'   Whether to clamp predictions to the range of the training target. Default is `TRUE`.
#'
#' The schedules, as functions of the fraction \eqn{t} of training steps already taken, are:
#' * `"coslog4"`: \eqn{(1 - \cos(2 \pi \log_2(1 + 15 t))) / 2}, four cycles of increasing length.
#' * `"flat_cos"`: one for \eqn{t < 0.5}, followed by a cosine decay to zero.
#' * `"cos"`: a cosine decay from one to zero.
#' * `"linear"`: \eqn{1 - t}.
#' * `"constant"`: one.
#'
#' @references
#' `r format_bib("holzmueller2024better")`
#' @export
LearnerTorchRealMLP = R6Class("LearnerTorchRealMLP",
  inherit = LearnerTorch,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(task_type, optimizer = NULL, loss = NULL, callbacks = list()) {
      assert_choice(task_type, c("classif", "regr"))
      tags = c("train", "required")
      param_set = ps(
        n_hidden_layers = p_int(lower = 0L, init = 3L, tags = tags),
        hidden_width = p_int(lower = 1L, init = 256L, tags = tags),
        act = p_fct(levels = c("selu", "mish", "relu", "silu", "gelu", "elu"),
          init = if (task_type == "classif") "selu" else "mish", tags = tags),
        use_parametric_act = p_lgl(init = TRUE, tags = tags),
        p_drop = p_dbl(lower = 0, upper = 1, init = 0.15, tags = tags),
        p_drop_sched = p_fct(levels = realmlp_schedules, init = "flat_cos", tags = tags),
        wd = p_dbl(lower = 0, init = 0.02, tags = tags),
        wd_sched = p_fct(levels = realmlp_schedules, init = "flat_cos", tags = tags),
        lr_sched = p_fct(levels = realmlp_schedules, init = "coslog4", tags = tags),
        bias_lr_factor = p_dbl(lower = 0, init = 0.1, tags = tags),
        act_lr_factor = p_dbl(lower = 0, init = 0.1, tags = tags),
        scale_lr_factor = p_dbl(lower = 0, init = 6, tags = tags),
        plr_lr_factor = p_dbl(lower = 0, init = 0.1, tags = tags),
        add_front_scale = p_lgl(init = TRUE, tags = tags),
        num_emb_type = p_fct(levels = c("pbld", "pblrd", "pl", "plr", "none"), init = "pbld", tags = tags),
        plr_sigma = p_dbl(lower = 0, init = 0.1, tags = tags),
        plr_hidden_1 = p_int(lower = 1L, init = 16L, tags = tags),
        plr_hidden_2 = p_int(lower = 1L, init = 4L, tags = tags),
        max_one_hot_cat_size = p_int(lower = 0L, init = 9L, tags = tags),
        embedding_size = p_int(lower = 1L, init = 8L, tags = tags)
      )
      param_set_task = if (task_type == "classif") {
        ps(ls_eps = p_dbl(lower = 0, upper = 1, init = 0.1, tags = tags))
      } else {
        ps(
          normalize_output = p_lgl(init = TRUE, tags = tags),
          clamp_output = p_lgl(init = TRUE, tags = tags)
        )
      }
      private$.param_set_base = ps_union(list(param_set, param_set_task))

      optimizer = optimizer %??%
        t_opt("adam", lr = if (task_type == "classif") 0.04 else 0.2, betas = c(0.9, 0.95))

      super$initialize(
        task_type = task_type,
        id = paste0(task_type, ".realmlp"),
        label = "RealMLP",
        param_set = alist(private$.param_set_base),
        optimizer = optimizer,
        callbacks = callbacks,
        loss = loss,
        man = "mlr3torch::mlr_learners.realmlp",
        feature_types = c("numeric", "integer", "logical", "factor", "ordered"),
        jittable = FALSE
      )
      private$.param_set_torch$set_values(
        epochs = 256L, batch_size = 256L, batch_size_predict = 1024L, drop_last = TRUE
      )
    }
  ),
  private = list(
    .ingress_tokens = function(task, param_vals) {
      n_num = n_num_features(task)
      n_categ = n_categ_features(task)
      if (n_num == 0L && n_categ == 0L) {
        stopf("Learner '%s' received task '%s' without any supported features.", self$id, task$id)
      }
      out = list()
      if (n_num > 0L) {
        out$x_num = ingress_num(shape = c(NA, n_num))
      }
      if (n_categ > 0L) {
        out$x_cat = ingress_categ(shape = c(NA, n_categ))
      }
      out
    },
    .network = function(task, param_vals) {
      # the column order must be that of the ingress tokens
      x_num = if (n_num_features(task) > 0L) {
        batchgetter_num(task$data(cols = ingress_num()$features(task)))
      }
      x_cat = if (n_categ_features(task) > 0L) {
        batchgetter_categ(task$data(cols = ingress_categ()$features(task)))
      }
      is_regr = task$task_type == "regr"
      nn_realmlp(
        x_num = x_num,
        x_cat = x_cat,
        cat_cardinalities = unname(categ_cardinalities(task)),
        d_out = if (is_regr) 1L else length(task$class_names),
        binary = !is_regr && "twoclass" %in% task$properties,
        classification = !is_regr,
        y = if (is_regr) task$truth(),
        normalize_output = is_regr && param_vals$normalize_output,
        clamp_output = is_regr && param_vals$clamp_output,
        n_hidden_layers = param_vals$n_hidden_layers,
        hidden_width = param_vals$hidden_width,
        act = param_vals$act,
        use_parametric_act = param_vals$use_parametric_act,
        p_drop = param_vals$p_drop * realmlp_schedule(param_vals$p_drop_sched, 0),
        add_front_scale = param_vals$add_front_scale,
        num_emb_type = param_vals$num_emb_type,
        plr_sigma = param_vals$plr_sigma,
        plr_hidden_1 = param_vals$plr_hidden_1,
        plr_hidden_2 = param_vals$plr_hidden_2,
        max_one_hot_cat_size = param_vals$max_one_hot_cat_size,
        embedding_size = param_vals$embedding_size,
        lr_factors = list(
          plr = param_vals$plr_lr_factor,
          scale = param_vals$scale_lr_factor,
          bias = param_vals$bias_lr_factor,
          act = param_vals$act_lr_factor
        )
      )
    },
    .setup_training = function(ctx, param_vals) {
      super$.setup_training(ctx, param_vals)
      schedulers = keep(ctx$callbacks, function(cb) inherits(cb, "CallbackSetLRScheduler"))
      if (length(schedulers)) {
        stopf("Learner '%s' schedules the learning rate itself (see the parameter 'lr_sched'), so it cannot be combined with the learning rate scheduler callback(s) %s.", # nolint
          self$id, paste0("'", names(schedulers), "'", collapse = ", "))
      }
      ctx$loss_fn = realmlp_wrap_loss(ctx$loss_fn, ctx$task_train, self$loss, param_vals, self$id)
      ctx$callbacks$realmlp = CallbackSetRealMLP$new(
        lr_sched = param_vals$lr_sched,
        wd = param_vals$wd,
        wd_sched = param_vals$wd_sched,
        p_drop = param_vals$p_drop,
        p_drop_sched = param_vals$p_drop_sched
      )
      ctx$optimizer = self$optimizer$generate(ctx$network$param_groups())
    }
  )
)

#' @include aaa.R
register_learner("classif.realmlp", LearnerTorchRealMLP)
register_learner("regr.realmlp", LearnerTorchRealMLP)
