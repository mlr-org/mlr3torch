# ======================================================================================
# TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling (ICLR 2025)
# ======================================================================================
#
# Upstream repository : https://github.com/yandex-research/tabm
# Ported from         : the official single-file package `tabm.py`, version 0.0.3,
#                       commit 28e47ae301c92ec37787dde1ce923a0793f405b4
#                       (the installed wheel `tabm==0.0.3` is byte-identical to `main`).
#                       `paper/bin/model.py` of the same commit was used as a cross
#                       reference for the `arch_type` semantics, the loss and the
#                       prediction aggregation.
# Upstream license    : Apache License 2.0
#                       (https://github.com/yandex-research/tabm/blob/main/LICENSE)
#
# Ported components (upstream name -> name here):
#   _init_rsqrt_uniform_        -> tabm_init_rsqrt_uniform_
#   _init_random_signs_         -> tabm_init_random_signs_
#   init_scaling_               -> tabm_init_scaling_
#   _OneHotEncoding             -> nn_tabm_one_hot
#   ElementwiseAffine           -> nn_tabm_elementwise_affine
#   ensemble_view / EnsembleView-> tabm_ensemble_view / nn_tabm_ensemble_view
#   LinearEnsemble              -> nn_tabm_linear_ensemble
#   LinearBatchEnsemble         -> nn_tabm_linear_batch_ensemble
#   MLPBackbone                 -> nn_tabm_mlp_backbone
#   MLPBackboneEnsemble         -> nn_tabm_mlp_backbone_ensemble
#   MLPBackboneMiniEnsemble     -> nn_tabm_mlp_backbone_mini_ensemble
#   MLPBackboneBatchEnsemble    -> nn_tabm_mlp_backbone_batch_ensemble
#   make_tabm_backbone          -> tabm_make_backbone
#   TabM                        -> nn_tabm
#
# Intentional deviations from upstream:
#
#  * `num_embeddings` (the `rtdl_num_embeddings` piecewise-linear / periodic embeddings
#    for numerical features) is NOT ported. Numerical features enter the backbone raw,
#    i.e. `d_features[i] == 1` for every numerical feature. Consequently the
#    `start_scaling_init = "normal"` default that upstream uses when embeddings are
#    present never applies, and `n_blocks` defaults to 3 (upstream's no-embedding
#    default) rather than 2.
#  * `share_training_batches = FALSE` is NOT supported: `forward()` only accepts
#    two-dimensional `x_num` / `x_cat`, so all k submodels always see the same batch.
#    (Upstream additionally accepts three-dimensional `(batch, k, d)` inputs.)
#  * `arch_type = "plain"` is added. It does not exist in the packaged `tabm.py`; it is
#    taken from `paper/bin/model.py` and denotes a plain (non-ensembled) MLP. As there,
#    its output is unsqueezed to `(batch, 1, d_out)` so that the loss, the prediction
#    aggregation and the output shape are uniform across all `arch_type`s.
#  * Categorical features are encoded with **1-based** integer codes (this is what
#    mlr3torch's `batchgetter_categ()` produces and what R torch's `nnf_one_hot()`
#    expects); upstream uses 0-based codes. `logical()` features are shifted by one by
#    `batchgetter_categ_tabm()` because `as.integer()` maps them to 0/1.
#  * The one-hot encoding is cast to float (as `paper/bin/model.py` does). The packaged
#    `tabm.py` keeps it as long, which makes purely categorical inputs fail for
#    `arch_type = "tabm-packed"` (the first operation is a matmul against a float
#    weight). This is a fix, not a behavioural change, for the other architectures.
#  * `activation` is restricted to a fixed set of names instead of upstream's
#    `getattr(torch.nn, activation)` lookup.
#  * The following upstream objects are NOT ported because `TabM` does not depend on
#    them: `BatchNorm1dEnsemble`, `LayerNormEnsemble`, the in-place layer replacement
#    helpers (`_replace_layers_`, `ensemble_linear_layers_`,
#    `batchensemble_linear_layers_`, `ensemble_batchnorm1d_layers_`,
#    `ensemble_layernorm_layers_`) and the `from_linear()` / `from_batchnorm1d()` /
#    `from_layernorm()` constructors.
#  * `TabM.make()` is not ported as a separate entry point; its (non-constant) default
#    values are the defaults of `nn_tabm()` and of the learner's `ParamSet`.
#  * `ensemble_view()` does not warn when a three-dimensional input is passed in eval
#    mode, because three-dimensional inputs are not supported here at all.
#  * The loss adapter (`nn_tabm_loss`) and the probability averaging in
#    `.encode_prediction()` are ported from `paper/bin/model.py` (the packaged
#    `tabm.py` contains the model only).
#
# ======================================================================================

# --------------------------------------------------------------------------------------
# Initialization (upstream section "Initialization")
# --------------------------------------------------------------------------------------

tabm_init_rsqrt_uniform_ = function(tensor, d) {
  d_rsqrt = d^(-0.5)
  nn_init_uniform_(tensor, -d_rsqrt, d_rsqrt)
}

tabm_init_random_signs_ = function(tensor) {
  with_no_grad(tensor$bernoulli_(0.5)$mul_(2)$add_(-1))
  tensor
}

# `distribution` is one of "ones", "normal", "random-signs".
# `chunks` (upstream: `chunks`) splits the last dimension of `x` into consecutive
# blocks; all values within one block share the same randomly drawn value.
tabm_init_scaling_ = function(x, distribution, chunks = NULL) {
  init_fn = switch(distribution,
    "ones" = nn_init_ones_,
    "normal" = nn_init_normal_,
    "random-signs" = tabm_init_random_signs_,
    stopf("Unknown scaling initialization '%s'.", distribution)
  )
  if (distribution == "ones" && !is.null(chunks)) {
    stopf("When the scaling initialization is 'ones', chunks must be NULL.")
  }
  if (is.null(chunks)) {
    return(init_fn(x))
  }
  shape = dim(x)
  if (sum(chunks) != shape[length(shape)]) {
    stopf("The tensor shape (%i) and the chunks (sum: %i) are incompatible.",
      shape[length(shape)], sum(chunks))
  }
  leading = shape[-length(shape)]
  with_no_grad({
    chunk_start = 0L
    for (chunk_size in chunks) {
      value = init_fn(do.call(torch_empty, as.list(c(leading, 1L))))
      x[.., (chunk_start + 1L):(chunk_start + chunk_size)] = value
      chunk_start = chunk_start + chunk_size
    }
  })
  x
}

# --------------------------------------------------------------------------------------
# Basic modules (upstream section "Basics modules")
# --------------------------------------------------------------------------------------

# Upstream: `_OneHotEncoding`.
# Deviation: the input codes are 1-based, i.e. the i-th feature takes values in
# `1:cardinalities[i]` (upstream: `0:(cardinalities[i] - 1)`).
nn_tabm_one_hot = nn_module("nn_tabm_one_hot",
  initialize = function(cardinalities) {
    self$cardinalities = assert_integerish(cardinalities, lower = 1L, any.missing = FALSE,
      min.len = 1L, coerce = TRUE)
  },
  forward = function(input) {
    cards = self$cardinalities
    torch_cat(lapply(seq_along(cards), function(i) {
      nnf_one_hot(input[, i], num_classes = cards[i])
    }), dim = -1L)$to(dtype = torch_float())
  }
)

# Upstream: `ElementwiseAffine`.
nn_tabm_elementwise_affine = nn_module("nn_tabm_elementwise_affine",
  initialize = function(shape, bias, scaling_init, scaling_init_chunks = NULL) {
    self$scaling_init = scaling_init
    self$scaling_init_chunks = scaling_init_chunks
    self$weight = nn_parameter(do.call(torch_empty, as.list(shape)))
    self$bias = if (bias) nn_parameter(do.call(torch_empty, as.list(shape))) else NULL
    self$reset_parameters()
  },
  reset_parameters = function() {
    tabm_init_scaling_(self$weight, self$scaling_init, self$scaling_init_chunks)
    if (!is.null(self$bias)) {
      nn_init_zeros_(self$bias)
    }
  },
  forward = function(input) {
    if (is.null(self$bias)) input * self$weight else input * self$weight + self$bias
  }
)

# --------------------------------------------------------------------------------------
# Ensemble modules (upstream section "Ensemble modules")
# --------------------------------------------------------------------------------------

# Upstream: `ensemble_view()`. Turns `(batch, d)` into `(batch, k, d)` without copying.
tabm_ensemble_view = function(x, k) {
  if (x$dim() != 2L) {
    stopf("The input must have two dimensions, but has %i.", x$dim())
  }
  x$unsqueeze(2L)$expand(c(-1L, k, -1L))
}

# Upstream: `EnsembleView`.
nn_tabm_ensemble_view = nn_module("nn_tabm_ensemble_view",
  initialize = function(k) {
    self$k = assert_int(k, lower = 1L, coerce = TRUE)
  },
  forward = function(input) {
    tabm_ensemble_view(input, self$k)
  }
)

# Upstream: `LinearEnsemble`. k independent linear layers applied to k inputs.
nn_tabm_linear_ensemble = nn_module("nn_tabm_linear_ensemble",
  initialize = function(in_features, out_features, bias = TRUE, k) {
    self$in_features = assert_int(in_features, lower = 1L, coerce = TRUE)
    self$out_features = assert_int(out_features, lower = 1L, coerce = TRUE)
    self$k = assert_int(k, lower = 1L, coerce = TRUE)
    self$weight = nn_parameter(torch_empty(self$k, self$in_features, self$out_features))
    self$bias = if (bias) nn_parameter(torch_empty(self$k, self$out_features)) else NULL
    self$reset_parameters()
  },
  reset_parameters = function() {
    tabm_init_rsqrt_uniform_(self$weight, self$in_features)
    if (!is.null(self$bias)) {
      tabm_init_rsqrt_uniform_(self$bias, self$in_features)
    }
  },
  forward = function(input) {
    x = input$transpose(1L, 2L)
    x = torch_matmul(x, self$weight)
    x = x$transpose(1L, 2L)
    if (!is.null(self$bias)) x = x + self$bias
    x
  }
)

# Upstream: `LinearBatchEnsemble`, i.e. equation (5) of the BatchEnsemble paper with the
# TabM-specific initialization options for the R and S matrices.
nn_tabm_linear_batch_ensemble = nn_module("nn_tabm_linear_batch_ensemble",
  initialize = function(in_features, out_features, bias = TRUE, k, scaling_init,
    first_scaling_init_chunks = NULL) {
    self$in_features = assert_int(in_features, lower = 1L, coerce = TRUE)
    self$out_features = assert_int(out_features, lower = 1L, coerce = TRUE)
    self$k = assert_int(k, lower = 1L, coerce = TRUE)
    assert_character(scaling_init, min.len = 1L, max.len = 2L, any.missing = FALSE)
    self$first_scaling_init = scaling_init[[1L]]
    self$second_scaling_init = scaling_init[[length(scaling_init)]]
    self$first_scaling_init_chunks = first_scaling_init_chunks

    self$weight = nn_parameter(torch_empty(self$out_features, self$in_features))
    self$r = nn_parameter(torch_empty(self$k, self$in_features))
    self$s = nn_parameter(torch_empty(self$k, self$out_features))
    self$bias = if (bias) nn_parameter(torch_empty(self$k, self$out_features)) else NULL
    self$reset_parameters()
  },
  reset_parameters = function() {
    tabm_init_rsqrt_uniform_(self$weight, self$in_features)
    tabm_init_scaling_(self$r, self$first_scaling_init, self$first_scaling_init_chunks)
    tabm_init_scaling_(self$s, self$second_scaling_init, NULL)
    if (!is.null(self$bias)) {
      # All k biases share the same initialization.
      bias_init = tabm_init_rsqrt_uniform_(torch_empty(self$out_features), self$in_features)
      with_no_grad(self$bias$copy_(bias_init$expand(c(self$k, self$out_features))))
    }
  },
  forward = function(input) {
    x = input * self$r
    x = torch_matmul(x, self$weight$t())
    x = x * self$s
    if (!is.null(self$bias)) x = x + self$bias
    x
  }
)

# --------------------------------------------------------------------------------------
# MLP backbones (upstream section "MLP modules")
# --------------------------------------------------------------------------------------

tabm_activation = function(activation) {
  switch(activation,
    "relu" = nn_relu(),
    "gelu" = nn_gelu(),
    "elu" = nn_elu(),
    "selu" = nn_selu(),
    "leaky_relu" = nn_leaky_relu(),
    "tanh" = nn_tanh(),
    "sigmoid" = nn_sigmoid(),
    stopf("Unknown activation '%s'.", activation)
  )
}

# Upstream: `_MLPBackboneBase.__init__()`. `make_linear(index, in_features, out_features)`
# is the `_make_linear()` hook of the respective subclass (index is 1-based here).
tabm_make_blocks = function(d_in, n_blocks, d_block, dropout, activation, make_linear) {
  assert_int(d_in, lower = 1L)
  assert_int(n_blocks, lower = 1L)
  assert_int(d_block, lower = 1L)
  nn_module_list(lapply(seq_len(n_blocks), function(i) {
    nn_sequential(
      make_linear(i, if (i == 1L) d_in else d_block, d_block),
      tabm_activation(activation),
      nn_dropout(dropout)
    )
  }))
}

# Upstream: `MLPBackbone`.
nn_tabm_mlp_backbone = nn_module("nn_tabm_mlp_backbone",
  initialize = function(d_in, n_blocks, d_block, dropout, activation = "relu") {
    self$n_blocks = n_blocks
    self$d_out = d_block
    self$blocks = tabm_make_blocks(d_in, n_blocks, d_block, dropout, activation,
      function(index, in_features, out_features) nn_linear(in_features, out_features))
  },
  forward = function(input) {
    x = input
    for (i in seq_len(self$n_blocks)) {
      x = self$blocks[[i]](x)
    }
    x
  }
)

# Upstream: `MLPBackboneEnsemble` (used by arch_type "tabm-packed").
nn_tabm_mlp_backbone_ensemble = nn_module("nn_tabm_mlp_backbone_ensemble",
  initialize = function(d_in, n_blocks, d_block, dropout, activation = "relu", k) {
    self$n_blocks = n_blocks
    self$k = k
    self$d_out = d_block
    self$blocks = tabm_make_blocks(d_in, n_blocks, d_block, dropout, activation,
      function(index, in_features, out_features) {
        nn_tabm_linear_ensemble(in_features, out_features, k = k)
      })
  },
  forward = function(input) {
    x = input
    for (i in seq_len(self$n_blocks)) {
      x = self$blocks[[i]](x)
    }
    x
  }
)

# Upstream: `MLPBackboneMiniEnsemble` (used by arch_type "tabm-mini").
nn_tabm_mlp_backbone_mini_ensemble = nn_module("nn_tabm_mlp_backbone_mini_ensemble",
  initialize = function(d_in, n_blocks, d_block, dropout, activation = "relu", k,
    affine_bias, affine_scaling_init, affine_scaling_init_chunks = NULL) {
    self$n_blocks = n_blocks
    self$k = k
    self$d_out = d_block
    self$blocks = tabm_make_blocks(d_in, n_blocks, d_block, dropout, activation,
      # The same linear layer is used by all k backbones.
      function(index, in_features, out_features) nn_linear(in_features, out_features))
    self$affine = nn_tabm_elementwise_affine(
      shape = c(k, d_in),
      bias = affine_bias,
      scaling_init = affine_scaling_init,
      scaling_init_chunks = affine_scaling_init_chunks
    )
  },
  forward = function(input) {
    x = self$affine(input)
    for (i in seq_len(self$n_blocks)) {
      x = self$blocks[[i]](x)
    }
    x
  }
)

# Upstream: `MLPBackboneBatchEnsemble` (used by arch_type "tabm").
nn_tabm_mlp_backbone_batch_ensemble = nn_module("nn_tabm_mlp_backbone_batch_ensemble",
  initialize = function(d_in, n_blocks, d_block, dropout, activation = "relu", k,
    tabm_init, scaling_init, start_scaling_init_chunks = NULL) {
    self$n_blocks = n_blocks
    self$k = k
    self$d_out = d_block
    self$blocks = tabm_make_blocks(d_in, n_blocks, d_block, dropout, activation,
      function(index, in_features, out_features) {
        nn_tabm_linear_batch_ensemble(
          in_features, out_features, k = k,
          scaling_init = if (tabm_init) {
            if (index == 1L) c(scaling_init, "ones") else "ones"
          } else {
            scaling_init
          },
          first_scaling_init_chunks = if (index == 1L) start_scaling_init_chunks else NULL
        )
      })
  },
  forward = function(input) {
    x = input
    for (i in seq_len(self$n_blocks)) {
      x = self$blocks[[i]](x)
    }
    x
  }
)

# Upstream: `make_tabm_backbone()`, extended with `arch_type = "plain"`.
tabm_make_backbone = function(d_in, n_blocks, d_block, dropout, activation, k, arch_type,
  start_scaling_init, start_scaling_init_chunks) {
  if (arch_type %in% c("tabm-packed", "plain")) {
    if (!is.null(start_scaling_init)) {
      stopf("When arch_type is '%s', start_scaling_init must be NULL.", arch_type)
    }
  } else if (is.null(start_scaling_init)) {
    stopf("When arch_type is '%s', start_scaling_init must not be NULL.", arch_type)
  }

  switch(arch_type,
    "tabm" = nn_tabm_mlp_backbone_batch_ensemble(
      d_in = d_in, n_blocks = n_blocks, d_block = d_block, dropout = dropout,
      activation = activation, k = k, tabm_init = TRUE,
      scaling_init = start_scaling_init,
      start_scaling_init_chunks = start_scaling_init_chunks
    ),
    "tabm-mini" = nn_tabm_mlp_backbone_mini_ensemble(
      d_in = d_in, n_blocks = n_blocks, d_block = d_block, dropout = dropout,
      activation = activation, k = k, affine_bias = FALSE,
      affine_scaling_init = start_scaling_init,
      affine_scaling_init_chunks = start_scaling_init_chunks
    ),
    "tabm-packed" = nn_tabm_mlp_backbone_ensemble(
      d_in = d_in, n_blocks = n_blocks, d_block = d_block, dropout = dropout,
      activation = activation, k = k
    ),
    "plain" = nn_tabm_mlp_backbone(
      d_in = d_in, n_blocks = n_blocks, d_block = d_block, dropout = dropout,
      activation = activation
    ),
    stopf("Unknown arch_type '%s'.", arch_type)
  )
}

# --------------------------------------------------------------------------------------
# The TabM module
# --------------------------------------------------------------------------------------

#' @title TabM Network
#'
#' @description
#' TabM -- a tabular deep learning model that makes **M**ultiple predictions.
#' One `nn_tabm` efficiently represents an ensemble of `k` MLPs that are trained in
#' parallel and that share most of their weights.
#'
#' Numerical features enter the network unchanged, categorical features are one-hot
#' encoded (their integer codes must be 1-based, which is what
#' [`batchgetter_categ()`] produces for `factor()` and `ordered()` features).
#' The concatenated flat representation is then processed by `k` (mostly shared) MLP
#' backbones.
#'
#' For an input of shape `(batch, n_num_features)` / `(batch, n_cat_features)` the output
#' shape is `(batch, k, d_out)`, i.e. one prediction per ensemble member.
#' For `arch_type = "plain"` the output shape is `(batch, 1, d_out)`.
#'
#' @section Ensemble Output:
#' Because the output contains the `k` predictions of the ensemble members, it can not be
#' fed into a standard loss function, and it must be aggregated before it can be
#' interpreted as a prediction.
#' [`LearnerTorchTabM`] (`lrn("classif.tabm")` / `lrn("regr.tabm")`) takes care of both.
#' When plugging `nn_tabm` into [`LearnerTorchModule`] (`lrn("classif.module")`) instead,
#' the loss must be wrapped in [`nn_tabm_loss()`], and the predictions of the resulting
#' learner are *not* aggregated over the `k` submodels.
#'
#' @param task ([`Task`][mlr3::Task] or `NULL`)\cr
#'   If provided, `n_num_features`, `cat_cardinalities` and `d_out` are inferred from
#'   the task (unless they are given explicitly). This makes it possible to use
#'   `nn_tabm` with [`LearnerTorchModule`].
#' @param n_num_features (`integer(1)`)\cr
#'   The number of numerical features.
#' @param cat_cardinalities (`integer()` or `NULL`)\cr
#'   The number of categories of each categorical feature.
#' @param d_out (`integer(1)` or `NULL`)\cr
#'   The output dimension. If `NULL`, the output of the `k` backbones is returned.
#' @param arch_type (`character(1)`)\cr
#'   One of `"tabm"` (default), `"tabm-mini"`, `"tabm-packed"` or `"plain"`.
#' @param k (`integer(1)`)\cr
#'   The number of ensemble members. Ignored for `arch_type = "plain"`.
#' @param n_blocks (`integer(1)`)\cr
#'   The number of blocks (depth) of the MLP backbone.
#' @param d_block (`integer(1)`)\cr
#'   The width of the MLP backbone.
#' @param dropout (`numeric(1)`)\cr
#'   The dropout rate.
#' @param activation (`character(1)`)\cr
#'   The activation function, one of `"relu"` (default), `"gelu"`, `"elu"`, `"selu"`,
#'   `"leaky_relu"`, `"tanh"` or `"sigmoid"`.
#' @param start_scaling_init (`character(1)` or `NULL`)\cr
#'   The initialization of the very first (non-shared) scaling, either `"random-signs"`
#'   or `"normal"`. Must be `NULL` for `arch_type` `"tabm-packed"` and `"plain"`.
#'   If `NULL` and the architecture requires it, `"random-signs"` is used.
#'
#' @references
#' `r format_bib("gorishniy2025tabm", "wen2020batchensemble")`
#'
#' @export
#' @examplesIf torch::torch_is_installed()
#' net = nn_tabm(n_num_features = 4, cat_cardinalities = c(3, 2), d_out = 3,
#'   k = 4, n_blocks = 2, d_block = 8, dropout = 0.1)
#' x_num = torch::torch_randn(5, 4)
#' x_cat = torch::torch_stack(list(
#'   torch::torch_randint(1, 3, 5, dtype = torch::torch_long()),
#'   torch::torch_randint(1, 2, 5, dtype = torch::torch_long())
#' ), dim = 2)
#' net(x_num = x_num, x_cat = x_cat)$shape
nn_tabm = nn_module("nn_tabm",
  initialize = function(task = NULL, n_num_features = NULL, cat_cardinalities = NULL,
    d_out = NULL, arch_type = "tabm", k = 32L, n_blocks = 3L, d_block = 512L,
    dropout = 0.1, activation = "relu", start_scaling_init = NULL) {
    if (!is.null(task)) {
      assert_class(task, "Task")
      if (is.null(n_num_features)) n_num_features = n_num_features(task)
      if (is.null(cat_cardinalities)) cat_cardinalities = tabm_cardinalities(task)
      if (is.null(d_out)) d_out = output_dim_for(task)
    }
    n_num_features = assert_int(n_num_features %??% 0L, lower = 0L, coerce = TRUE)
    cat_cardinalities = assert_integerish(cat_cardinalities %??% integer(0),
      lower = 1L, any.missing = FALSE, coerce = TRUE)
    d_out = assert_int(d_out, lower = 1L, null.ok = TRUE, coerce = TRUE)
    arch_type = assert_choice(arch_type, c("tabm", "tabm-mini", "tabm-packed", "plain"))
    k = assert_int(k, lower = 1L, coerce = TRUE)
    assert_number(dropout, lower = 0, upper = 1)
    assert_choice(start_scaling_init, c("random-signs", "normal"), null.ok = TRUE)

    if (n_num_features == 0L && !length(cat_cardinalities)) {
      stopf("nn_tabm() requires at least one numerical or one categorical feature.")
    }

    # Representation sizes of all features (upstream: `d_features`), which double as the
    # initialization chunks of the very first scaling.
    d_features = c(rep(1L, n_num_features), cat_cardinalities)

    self$n_num_features = n_num_features
    self$n_cat_features = length(cat_cardinalities)
    self$arch_type = arch_type
    self$cat_module = if (length(cat_cardinalities)) nn_tabm_one_hot(cat_cardinalities) else NULL

    if (arch_type %in% c("tabm-packed", "plain")) {
      if (!is.null(start_scaling_init)) {
        stopf("When arch_type is '%s', start_scaling_init must be NULL.", arch_type)
      }
    } else {
      # Upstream `TabM.make()` uses "random-signs" when there are no num_embeddings.
      start_scaling_init = start_scaling_init %??% "random-signs"
    }

    self$k = if (arch_type == "plain") 1L else k
    self$ensemble_view = if (arch_type == "plain") NULL else nn_tabm_ensemble_view(k = k)
    self$backbone = tabm_make_backbone(
      d_in = sum(d_features), n_blocks = n_blocks, d_block = d_block, dropout = dropout,
      activation = activation, k = k, arch_type = arch_type,
      start_scaling_init = start_scaling_init,
      start_scaling_init_chunks = if (is.null(start_scaling_init)) NULL else d_features
    )
    self$output = if (is.null(d_out)) {
      NULL
    } else if (arch_type == "plain") {
      nn_linear(self$backbone$d_out, d_out)
    } else {
      nn_tabm_linear_ensemble(self$backbone$d_out, d_out, k = k)
    }
  },
  forward = function(x_num = NULL, x_cat = NULL) {
    if (self$n_num_features > 0L && is.null(x_num)) {
      stopf("nn_tabm was built with %i numerical features, but x_num is NULL.", self$n_num_features)
    }
    if (self$n_cat_features > 0L && is.null(x_cat)) {
      stopf("nn_tabm was built with %i categorical features, but x_cat is NULL.", self$n_cat_features)
    }
    parts = list()
    if (!is.null(x_num)) parts[[length(parts) + 1L]] = x_num
    if (!is.null(x_cat)) parts[[length(parts) + 1L]] = self$cat_module(x_cat)
    x = if (length(parts) == 1L) parts[[1L]] else torch_cat(parts, dim = 2L)

    if (is.null(self$ensemble_view)) {
      # arch_type "plain": (B, D) -> (B, D_OUT) -> (B, 1, D_OUT)
      x = self$backbone(x)
      if (!is.null(self$output)) x = self$output(x)
      x$unsqueeze(2L)
    } else {
      x = self$ensemble_view(x)
      x = self$backbone(x)
      if (!is.null(self$output)) x = self$output(x)
      x
    }
  }
)

# --------------------------------------------------------------------------------------
# Loss adapter
# --------------------------------------------------------------------------------------

# Upstream: `loss_fn()` in `paper/bin/model.py`.
#' @title Ensemble Loss Adapter for TabM
#'
#' @description
#' Adapts a loss function to the `(batch, k, d_out)` output of [`nn_tabm()`]:
#' the ensemble dimension is folded into the batch dimension and every target is repeated
#' `k` times, so that all `k` submodels are trained on the full batch.
#' This is what the reference implementation of TabM does.
#'
#' [`LearnerTorchTabM`] applies this adapter automatically to whatever loss it is
#' configured with; it only needs to be used explicitly when combining [`nn_tabm()`] with
#' [`LearnerTorchModule`].
#'
#' @param loss ([`nn_module`][torch::nn_module])\cr
#'   The loss module that is applied to the folded prediction and the repeated target.
#'
#' @references
#' `r format_bib("gorishniy2025tabm")`
#'
#' @export
#' @examplesIf torch::torch_is_installed()
#' loss = nn_tabm_loss(torch::nn_cross_entropy_loss())
#' input = torch::torch_randn(6, 4, 3)
#' target = torch::torch_randint(1, 3, 6, dtype = torch::torch_long())
#' loss(input, target)
nn_tabm_loss = nn_module("nn_tabm_loss",
  initialize = function(loss) {
    self$loss = loss
  },
  forward = function(input, target) {
    k = input$shape[2L]
    self$loss(
      input$flatten(start_dim = 1L, end_dim = 2L),
      target$repeat_interleave(k, dim = 1L)
    )
  }
)

# Wraps a `TorchLoss` so that the generated loss module folds the ensemble dimension.
tabm_wrap_loss = function(loss) {
  loss = as_torch_loss(loss, clone = TRUE)
  if (isTRUE(attr(loss$generator, "tabm_wrapped"))) {
    return(loss)
  }
  inner = loss$generator
  needs_task = is.function(inner) && "task" %in% formalArgs(inner)
  generator = function(task, ...) {
    args = list(...)
    if (needs_task) args$task = task
    nn_tabm_loss(invoke(inner, .args = args))
  }
  attr(generator, "tabm_wrapped") = TRUE
  TorchLoss$new(
    torch_loss = generator,
    task_types = loss$task_types,
    param_set = loss$param_set,
    id = loss$id,
    label = loss$label,
    packages = loss$packages,
    man = loss$man
  )
}

# --------------------------------------------------------------------------------------
# The learner
# --------------------------------------------------------------------------------------

# The categorical cardinalities in the column order produced by
# `selector_type(c("factor", "ordered", "logical"))`, i.e. in task feature order.
# `Task$levels()` returns `NULL` for logical features, whose cardinality is 2.
tabm_cardinalities = function(task) {
  categ = task$feature_names[task$feature_types$type %in% c("factor", "ordered", "logical")]
  if (!length(categ)) {
    return(integer(0))
  }
  cardinalities = lengths(task$levels(categ))[categ]
  cardinalities[cardinalities == 0L] = 2L
  as.integer(cardinalities)
}

#' @title Batchgetter for Categorical Data (1-based)
#' @description
#' Like [`batchgetter_categ()`], but shifts `logical()` columns by one so that *all*
#' columns contain 1-based integer codes (`as.integer()` maps `logical()` to `0`/`1`,
#' but to `1:n` for `factor()`).
#' @param data (`data.table`)\cr
#'   `data.table` to be converted to a `tensor`.
#' @param ... (any)\cr
#'   Unused.
#' @noRd
batchgetter_categ_tabm = function(data, ...) {
  torch_tensor(
    data = as.matrix(data[, lapply(.SD, function(x) if (is.logical(x)) as.integer(x) + 1L else as.integer(x))]),
    dtype = torch_long()
  )
}

#' @title TabM
#'
#' @templateVar name tabm
#' @templateVar task_types classif, regr
#' @templateVar param_vals k = 4, n_blocks = 2, d_block = 32
#' @template params_learner
#' @template learner
#' @template learner_example
#'
#' @description
#' TabM is an MLP-based tabular deep learning model that efficiently represents an
#' ensemble of `k` MLPs: the `k` submodels are trained in parallel on the same batches
#' and share most of their weights, which acts as a strong regularizer.
#' The network produces `k` predictions per observation; the learner averages the
#' predicted *probabilities* (classification) or the predicted values (regression)
#' over the `k` submodels, and its loss function trains all `k` submodels jointly.
#'
#' Numerical features are used as-is, categorical features are one-hot encoded.
#' The piecewise-linear / periodic numerical embeddings of the reference implementation
#' are not available.
#'
#' @section Parameters:
#' Parameters from [`LearnerTorch`], as well as:
#' * `arch_type` :: `character(1)`\cr
#'   The architecture type, one of:
#'   * `"tabm"` (default) -- BatchEnsemble with the TabM initialization, i.e. all
#'     multiplicative adapters except the very first one are initialized with ones.
#'   * `"tabm-mini"` -- all non-shared parameters are concentrated in a single
#'     elementwise affine transformation applied to the input.
#'   * `"tabm-packed"` -- `k` fully independent (packed) MLPs.
#'   * `"plain"` -- a plain MLP without any ensembling (`k` is ignored).
#' * `k` :: `integer(1)`\cr
#'   The number of ensemble members. Default is `32`.
#' * `n_blocks` :: `integer(1)`\cr
#'   The number of blocks of the MLP backbone. Default is `3`.
#' * `d_block` :: `integer(1)`\cr
#'   The width of the MLP backbone. Default is `512`.
#' * `dropout` :: `numeric(1)`\cr
#'   The dropout rate. Default is `0.1`.
#' * `activation` :: `character(1)`\cr
#'   The activation function. Default is `"relu"`.
#' * `start_scaling_init` :: `character(1)`\cr
#'   The initialization of the very first (non-shared) scaling, either `"random-signs"`
#'   (default) or `"normal"`. Ignored for `arch_type` `"tabm-packed"` and `"plain"`.
#'
#' @section Loss and Prediction:
#' The network output has shape `(batch, k, d_out)`.
#' The learner therefore wraps the configured loss function (the default, or one passed
#' via the `loss` construction argument) such that the ensemble dimension is folded into
#' the batch dimension and each target is repeated `k` times.
#' For prediction, the per-submodel probabilities (softmax for multiclass, sigmoid for
#' binary) are averaged over the `k` submodels; for regression the outputs are averaged.
#'
#' @references
#' `r format_bib("gorishniy2025tabm", "wen2020batchensemble")`
#' @export
LearnerTorchTabM = R6Class("LearnerTorchTabM",
  inherit = LearnerTorch,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(task_type, optimizer = NULL, loss = NULL, callbacks = list()) {
      private$.param_set_base = ps(
        arch_type = p_fct(levels = c("tabm", "tabm-mini", "tabm-packed", "plain"),
          init = "tabm", tags = "train"),
        k = p_int(lower = 1L, init = 32L, tags = "train"),
        n_blocks = p_int(lower = 1L, init = 3L, tags = "train"),
        d_block = p_int(lower = 1L, init = 512L, tags = "train"),
        dropout = p_dbl(lower = 0, upper = 1, init = 0.1, tags = "train"),
        activation = p_fct(
          levels = c("relu", "gelu", "elu", "selu", "leaky_relu", "tanh", "sigmoid"),
          init = "relu", tags = "train"),
        start_scaling_init = p_fct(levels = c("random-signs", "normal"),
          init = "random-signs", tags = "train")
      )

      if (is.null(loss)) {
        loss = t_loss(switch(task_type, classif = "cross_entropy", regr = "mse"))
      }

      super$initialize(
        task_type = task_type,
        id = paste0(task_type, ".tabm"),
        label = "TabM",
        param_set = alist(private$.param_set_base),
        optimizer = optimizer,
        callbacks = callbacks,
        loss = loss,
        man = "mlr3torch::mlr_learners.tabm",
        feature_types = c("numeric", "integer", "logical", "factor", "ordered"),
        jittable = TRUE
      )
    }
  ),
  active = list(
    #' @field loss ([`TorchLoss`])\cr
    #' The torch loss. Whatever loss is assigned is wrapped such that the ensemble
    #' dimension of the network output is folded into the batch dimension, see section
    #' *Loss and Prediction*.
    loss = function(rhs) {
      if (!missing(rhs)) {
        private$.param_set = NULL
        loss = tabm_wrap_loss(rhs)
        assert_choice(self$task_type, loss$task_types)
        private$.loss = loss
        self$packages = unique(c(self$packages, loss$packages))
      }
      private$.loss
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
        out$x_cat = TorchIngressToken(
          features = selector_type(c("factor", "ordered", "logical")),
          batchgetter = batchgetter_categ_tabm,
          shape = c(NA, n_categ)
        )
      }
      out
    },
    .network = function(task, param_vals) {
      arch_type = param_vals$arch_type %??% "tabm"
      nn_tabm(
        n_num_features = n_num_features(task),
        cat_cardinalities = tabm_cardinalities(task),
        d_out = output_dim_for(task),
        arch_type = arch_type,
        k = param_vals$k %??% 32L,
        n_blocks = param_vals$n_blocks %??% 3L,
        d_block = param_vals$d_block %??% 512L,
        dropout = param_vals$dropout %??% 0.1,
        activation = param_vals$activation %??% "relu",
        # upstream requires this to be NULL for these architectures
        start_scaling_init = if (arch_type %in% c("tabm-packed", "plain")) {
          NULL
        } else {
          param_vals$start_scaling_init %??% "random-signs"
        }
      )
    },
    # The network returns one prediction per submodel, i.e. a tensor of shape
    # (batch, k, d_out). Upstream averages the *probabilities* of the k submodels
    # (see `paper/bin/model.py`), which is not the same as averaging the logits.
    # `encode_prediction_default()` expects scores, so the averaged probabilities are
    # mapped back to the score scale (log / logit); this roundtrip is exact up to
    # floating point accuracy because softmax(log(p)) == p and sigmoid(logit(p)) == p.
    .encode_prediction = function(predict_tensor, task) {
      reduced = with_no_grad({
        if (task$task_type == "regr") {
          predict_tensor$mean(dim = 2L)
        } else if ("twoclass" %in% task$properties) {
          p = torch_sigmoid(predict_tensor)$mean(dim = 2L)$clamp(min = 1e-7, max = 1 - 1e-7)
          torch_log(p) - torch_log1p(-p)
        } else {
          p = nnf_softmax(predict_tensor, dim = 3L)$mean(dim = 2L)
          torch_log(p$clamp(min = 1e-30))
        }
      })
      encode_prediction_default(
        predict_tensor = reduced,
        predict_type = self$predict_type,
        task = task
      )
    }
  )
)

#' @include aaa.R
register_learner("classif.tabm", LearnerTorchTabM)
register_learner("regr.tabm", LearnerTorchTabM)
