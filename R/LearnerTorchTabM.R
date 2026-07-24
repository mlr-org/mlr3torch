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
#   MLPBackboneEnsemble         -> nn_tabm_mlp_backbone_ensemble
#   MLPBackboneMiniEnsemble     -> nn_tabm_mlp_backbone_mini_ensemble
#   MLPBackboneBatchEnsemble    -> nn_tabm_mlp_backbone_batch_ensemble
#   make_tabm_backbone          -> tabm_make_backbone
#   TabM                        -> nn_tabm
#
# The `num_embeddings` modules live in `R/num_embeddings.R` (a separate port of the
# `rtdl_num_embeddings` package, MIT; see the header of that file).
#
# Intentional deviations from upstream:
#
#  * `share_training_batches = FALSE` is NOT supported: `forward()` only accepts
#    two-dimensional `x_num` / `x_cat`, so all k submodels always see the same batch.
#    (Upstream additionally accepts three-dimensional `(batch, k, d)` inputs.)
#  * Categorical features are encoded with **1-based** integer codes (this is what
#    mlr3torch's `batchgetter_categ()` produces and what R torch's `nnf_one_hot()`
#    expects); upstream uses 0-based codes.
#  * The one-hot encoding is cast to float. Upstream's `_OneHotEncoding.forward()`
#    returns a `long` tensor and `TabM.forward()` never casts it; when numerical
#    features are present, `torch.column_stack()` implicitly promotes the concatenation
#    to float, so the cast is a no-op there. For purely categorical input, however,
#    upstream genuinely fails ("expected scalar type Long but found Float") for
#    `arch_type = "tabm-packed"`, whose first operation is a matmul against a float
#    weight. Casting unconditionally (which is what `paper/bin/model.py` does) is
#    therefore a fix for the categorical-only case and a no-op everywhere else.
#  * `activation` accepts an `nn_module_generator`, any function returning an
#    `nn_module`, or a name resolved against the `torch` package, instead of upstream's
#    `getattr(torch.nn, activation)` lookup.
#  * The following upstream objects are NOT ported because `TabM` does not depend on
#    them: `BatchNorm1dEnsemble`, `LayerNormEnsemble`, `MLPBackbone` (the non-ensembled
#    backbone, which `make_tabm_backbone()` never constructs), the in-place layer
#    replacement helpers (`_replace_layers_`, `ensemble_linear_layers_`,
#    `batchensemble_linear_layers_`, `ensemble_batchnorm1d_layers_`,
#    `ensemble_layernorm_layers_`) and the `from_linear()` / `from_batchnorm1d()` /
#    `from_layernorm()` constructors.
#  * `TabM.make()` is not ported as a separate entry point; its default values (which
#    depend on whether `num_embeddings` is used) are implemented by `nn_tabm()` and by
#    the learner's `.network()`.
#  * `nn_tabm()` accepts any `nn_module` with a `get_output_shape()` method as
#    `num_embeddings`, whereas upstream only accepts `LinearReLUEmbeddings`,
#    `PeriodicEmbeddings` and `PiecewiseLinearEmbeddings`. The check that
#    piecewise-linear embeddings use `version = "B"` is kept.
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

# Upstream resolves `activation` via `getattr(torch.nn, activation)`. Here, in addition
# to a name, a module generator (e.g. `nn_relu`) or any function returning an `nn_module`
# is accepted. A *fresh* module is constructed on every call, because each block needs
# its own activation instance.
tabm_activation = function(activation) {
  if (is.function(activation)) {
    module = activation()
    if (!inherits(module, "nn_module")) {
      stopf("The `activation` function must return an `nn_module`, but it returned an object of class '%s'.", class(module)[[1L]]) # nolint
    }
    return(module)
  }
  if (!test_string(activation, min.chars = 1L)) {
    stopf("`activation` must be a `character(1)`, an `nn_module_generator` or a function returning an `nn_module`, but is of class '%s'.", class(activation)[[1L]]) # nolint
  }
  ns = asNamespace("torch")
  get_generator = function(nm) {
    if (!exists(nm, envir = ns, inherits = FALSE)) {
      return(NULL)
    }
    generator = get(nm, envir = ns)
    if (inherits(generator, "nn_module_generator")) generator else NULL
  }
  # fast path: "nn_relu", "relu", "ReLU"
  for (nm in unique(c(activation, paste0("nn_", activation), paste0("nn_", tolower(activation))))) {
    generator = get_generator(nm)
    if (!is.null(generator)) {
      return(tabm_activation(generator))
    }
  }
  # slow path: match the torch.nn spelling, e.g. "LeakyReLU" -> `nn_leaky_relu`
  normalize = function(x) sub("^nn", "", gsub("[^a-z0-9]", "", tolower(x)))
  target = normalize(activation)
  for (nm in ls(ns, pattern = "^nn_")) {
    if (normalize(nm) == target) {
      generator = get_generator(nm)
      if (!is.null(generator)) {
        return(tabm_activation(generator))
      }
    }
  }
  stopf("Cannot resolve the activation '%s'. Provide the name of an activation of the torch package (e.g. \"relu\", \"nn_relu\" or \"ReLU\"), an `nn_module_generator` (e.g. `nn_relu`), or a function returning an `nn_module`.", activation) # nolint
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

# Upstream: `make_tabm_backbone()`.
tabm_make_backbone = function(d_in, n_blocks, d_block, dropout, activation, k, arch_type,
  start_scaling_init, start_scaling_init_chunks) {
  if (arch_type == "tabm-packed") {
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
#' Numerical features enter the network unchanged, or, if `num_embeddings` is given, are
#' first embedded feature-wise (see [`nn_linear_relu_embeddings()`],
#' [`nn_periodic_embeddings()`] and [`nn_piecewise_linear_embeddings()`]).
#' Categorical features are one-hot encoded (their integer codes must be 1-based, which
#' is what [`batchgetter_categ()`] produces).
#' The concatenated flat representation is then processed by `k` (mostly shared) MLP
#' backbones.
#'
#' For an input of shape `(batch, n_num_features)` / `(batch, n_cat_features)` the output
#' shape is `(batch, k, d_out)`, i.e. one prediction per ensemble member.
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
#' @param num_embeddings ([`nn_module`][torch::nn_module] or `NULL`)\cr
#'   Embeddings for the numerical features, applied before the backbone and shared
#'   between the `k` submodels. Must provide a `get_output_shape()` method returning
#'   `c(n_num_features, d_embedding)`; [`nn_linear_relu_embeddings()`],
#'   [`nn_periodic_embeddings()`] and [`nn_piecewise_linear_embeddings()`] (with
#'   `version = "B"`) do. If `NULL` (default), the numerical features enter the backbone
#'   unchanged.
#' @param arch_type (`character(1)`)\cr
#'   One of `"tabm"` (default), `"tabm-mini"` or `"tabm-packed"`.
#' @param k (`integer(1)`)\cr
#'   The number of ensemble members.
#' @param n_blocks (`integer(1)`)\cr
#'   The number of blocks (depth) of the MLP backbone.
#'   If `NULL`, `2` is used when `num_embeddings` is given and `3` otherwise.
#' @param d_block (`integer(1)`)\cr
#'   The width of the MLP backbone.
#' @param dropout (`numeric(1)`)\cr
#'   The dropout rate.
#' @param activation (`character(1)`, `nn_module_generator` or `function`)\cr
#'   The activation function. Either the name of an activation of the `torch` package
#'   (e.g. `"relu"`, `"nn_relu"` or `"ReLU"`), an
#'   [`nn_module_generator`][torch::nn_module] such as [`nn_relu`][torch::nn_relu], or a
#'   function returning an [`nn_module`][torch::nn_module]. Default is `"relu"`.
#' @param start_scaling_init (`character(1)` or `NULL`)\cr
#'   The initialization of the very first (non-shared) scaling, either `"random-signs"`
#'   or `"normal"`. Must be `NULL` for `arch_type = "tabm-packed"`.
#'   If `NULL` otherwise, `"normal"` is used when `num_embeddings` is given and
#'   `"random-signs"` otherwise (this is upstream's `TabM.make()` heuristic).
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
#'
#' # with periodic embeddings for the numerical features
#' net = nn_tabm(n_num_features = 4, d_out = 3, k = 4, n_blocks = 2, d_block = 8,
#'   num_embeddings = nn_periodic_embeddings(4, d_embedding = 6, lite = FALSE))
#' net(x_num = x_num)$shape
nn_tabm = nn_module("nn_tabm",
  initialize = function(task = NULL, n_num_features = NULL, cat_cardinalities = NULL,
    d_out = NULL, num_embeddings = NULL, arch_type = "tabm", k = 32L, n_blocks = NULL,
    d_block = 512L, dropout = 0.1, activation = "relu", start_scaling_init = NULL) {
    if (!is.null(task)) {
      assert_class(task, "Task")
      if (is.null(n_num_features)) n_num_features = n_num_features(task)
      if (is.null(cat_cardinalities)) cat_cardinalities = unname(categ_cardinalities(task))
      if (is.null(d_out)) d_out = output_dim_for(task)
    }
    n_num_features = assert_int(n_num_features %??% 0L, lower = 0L, coerce = TRUE)
    cat_cardinalities = assert_integerish(cat_cardinalities %??% integer(0),
      lower = 1L, any.missing = FALSE, coerce = TRUE)
    d_out = assert_int(d_out, lower = 1L, null.ok = TRUE, coerce = TRUE)
    arch_type = assert_choice(arch_type, c("tabm", "tabm-mini", "tabm-packed"))
    k = assert_int(k, lower = 1L, coerce = TRUE)
    assert_number(dropout, lower = 0, upper = 1)
    assert_choice(start_scaling_init, c("random-signs", "normal"), null.ok = TRUE)
    assert_class(num_embeddings, "nn_module", null.ok = TRUE)

    if (n_num_features == 0L && !length(cat_cardinalities)) {
      stopf("nn_tabm() requires at least one numerical or one categorical feature.")
    }

    # Representation sizes of all features (upstream: `d_features`), which double as the
    # initialization chunks of the very first scaling.
    d_features = if (is.null(num_embeddings)) {
      rep(1L, n_num_features)
    } else {
      if (n_num_features == 0L) {
        stopf("nn_tabm() received `num_embeddings`, but there are no numerical features.")
      }
      if (!is.function(num_embeddings$get_output_shape)) {
        stopf("The `num_embeddings` module must provide a `get_output_shape()` method.")
      }
      shape = num_embeddings$get_output_shape()
      if (shape[[1L]] != n_num_features) {
        stopf("The `num_embeddings` module was created for %i features, but n_num_features is %i.", shape[[1L]], n_num_features) # nolint
      }
      if (inherits(num_embeddings, "nn_piecewise_linear_embeddings") && is.null(num_embeddings$linear0)) { # nolint
        stopf("When using nn_piecewise_linear_embeddings() as `num_embeddings`, set version = \"B\".") # nolint
      }
      rep(as.integer(shape[[2L]]), n_num_features)
    }
    d_features = c(d_features, cat_cardinalities)

    self$n_num_features = n_num_features
    self$n_cat_features = length(cat_cardinalities)
    self$arch_type = arch_type
    self$num_module = num_embeddings
    self$cat_module = if (length(cat_cardinalities)) nn_tabm_one_hot(cat_cardinalities) else NULL

    if (arch_type == "tabm-packed") {
      if (!is.null(start_scaling_init)) {
        stopf("When arch_type is '%s', start_scaling_init must be NULL.", arch_type)
      }
    } else {
      # Upstream `TabM.make()`: "normal" if there are non-trivial modules before the
      # backbone (i.e. num_embeddings), "random-signs" otherwise.
      start_scaling_init = start_scaling_init %??%
        if (is.null(num_embeddings)) "random-signs" else "normal"
    }
    # Upstream `TabM.make()`: 2 blocks with embeddings, 3 without.
    n_blocks = n_blocks %??% if (is.null(num_embeddings)) 3L else 2L

    self$k = k
    self$ensemble_view = nn_tabm_ensemble_view(k = k)
    self$backbone = tabm_make_backbone(
      d_in = sum(d_features), n_blocks = n_blocks, d_block = d_block, dropout = dropout,
      activation = activation, k = k, arch_type = arch_type,
      start_scaling_init = start_scaling_init,
      start_scaling_init_chunks = if (is.null(start_scaling_init)) NULL else d_features
    )
    self$output = if (is.null(d_out)) {
      NULL
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
    if (!is.null(x_num)) {
      x_n = if (is.null(self$num_module)) x_num else self$num_module(x_num)
      # (B, n_num, d_embedding) -> (B, n_num * d_embedding); a no-op without embeddings
      parts[[length(parts) + 1L]] = x_n$flatten(start_dim = 2L)
    }
    if (!is.null(x_cat)) parts[[length(parts) + 1L]] = self$cat_module(x_cat)
    x = if (length(parts) == 1L) parts[[1L]] else torch_cat(parts, dim = 2L)

    x = self$ensemble_view(x)
    x = self$backbone(x)
    if (!is.null(self$output)) x = self$output(x)
    x
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

# Build the `num_embeddings` module from the learner's parameter values.
# The defaults follow the official TabM usage example
# (https://github.com/yandex-research/tabm/blob/main/example.ipynb):
#   LinearReLUEmbeddings(n)                                   -> d_embedding = 32 (upstream default)
#   PeriodicEmbeddings(n, lite = FALSE)                       -> d_embedding = 24 (upstream default)
#   PiecewiseLinearEmbeddings(bins, 16, activation = FALSE, version = "B")
tabm_make_num_embeddings = function(type, n_num_features, param_vals, x_num = NULL) {
  if (is.null(type) || identical(type, "none")) {
    return(NULL)
  }
  if (n_num_features == 0L) {
    stopf("The parameter 'num_embeddings' is set to '%s', but the task has no numerical features.", type)
  }
  switch(type,
    "linear_relu" = nn_linear_relu_embeddings(
      n_num_features,
      d_embedding = param_vals$d_embedding %??% 32L
    ),
    "periodic" = nn_periodic_embeddings(
      n_num_features,
      d_embedding = param_vals$d_embedding %??% 24L,
      n_frequencies = param_vals$n_frequencies %??% 48L,
      frequency_init_scale = param_vals$frequency_init_scale %??% 0.01,
      activation = param_vals$embedding_activation %??% TRUE,
      lite = param_vals$lite %??% FALSE
    ),
    "piecewise_linear" = {
      n_bins = param_vals$n_bins %??% 48L
      bins = tryCatch(compute_bins(x_num, n_bins = n_bins), error = function(e) {
        stopf("Cannot compute the bins for the piecewise-linear embeddings (n_bins = %i): %s", n_bins, conditionMessage(e)) # nolint
      })
      nn_piecewise_linear_embeddings(
        bins,
        d_embedding = param_vals$d_embedding %??% 16L,
        activation = param_vals$embedding_activation %??% FALSE,
        # TabM requires version "B"
        version = "B"
      )
    },
    stopf("Unknown num_embeddings type '%s'.", type)
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
#' Numerical features are used as-is, or -- if the `num_embeddings` parameter is set --
#' embedded feature-wise first, which usually improves the performance considerably.
#' Categorical features are one-hot encoded.
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
#' * `k` :: `integer(1)`\cr
#'   The number of ensemble members. Default is `32`.
#' * `n_blocks` :: `integer(1)`\cr
#'   The number of blocks of the MLP backbone.
#'   If unset, `2` is used when `num_embeddings` is set and `3` otherwise.
#' * `d_block` :: `integer(1)`\cr
#'   The width of the MLP backbone. Default is `512`.
#' * `dropout` :: `numeric(1)`\cr
#'   The dropout rate. Default is `0.1`.
#' * `activation` :: `character(1)`, `nn_module_generator` or `function`\cr
#'   The activation function of the MLP backbone. Either the name of an activation of
#'   the `torch` package (e.g. `"relu"`, `"nn_relu"` or `"ReLU"`), an
#'   [`nn_module_generator`][torch::nn_module] such as [`nn_relu`][torch::nn_relu], or a
#'   function returning an [`nn_module`][torch::nn_module]. Default is `"relu"`.
#' * `start_scaling_init` :: `character(1)`\cr
#'   The initialization of the very first (non-shared) scaling, either `"random-signs"`
#'   or `"normal"`. Ignored for `arch_type = "tabm-packed"`. If unset, `"normal"` is
#'   used when `num_embeddings` is set and `"random-signs"` otherwise.
#'
#' Parameters of the embeddings for the numerical features:
#' * `num_embeddings` :: `character(1)`\cr
#'   The type of the numerical feature embeddings, one of `"none"` (default),
#'   `"linear_relu"` ([`nn_linear_relu_embeddings()`]), `"periodic"`
#'   ([`nn_periodic_embeddings()`]) or `"piecewise_linear"`
#'   ([`nn_piecewise_linear_embeddings()`] with `version = "B"`, the variant required by
#'   TabM). The last two usually perform best.
#' * `d_embedding` :: `integer(1)`\cr
#'   The embedding size. If unset, `32` is used for `"linear_relu"`, `24` for
#'   `"periodic"` and `16` for `"piecewise_linear"`.
#' * `n_frequencies` :: `integer(1)`\cr
#'   `"periodic"` only: the number of frequencies per feature. Default is `48`.
#' * `frequency_init_scale` :: `numeric(1)`\cr
#'   `"periodic"` only: the initialization scale of the frequencies. This is an
#'   important hyperparameter. Default is `0.01`.
#' * `lite` :: `logical(1)`\cr
#'   `"periodic"` only: whether the outer linear layer is shared between all features.
#'   Default is `FALSE`.
#' * `embedding_activation` :: `logical(1)`\cr
#'   `"periodic"` and `"piecewise_linear"` only: whether a ReLU is applied at the end of
#'   the embedding. If unset, `TRUE` is used for `"periodic"` and `FALSE` for
#'   `"piecewise_linear"`.
#' * `n_bins` :: `integer(1)`\cr
#'   `"piecewise_linear"` only: the number of quantile bins, computed from the training
#'   data with [`compute_bins()`]. Must be smaller than the number of training
#'   observations. Default is `48`.
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
      check_activation = crate(function(x) {
        if (test_string(x, min.chars = 1L) || is.function(x)) {
          return(TRUE)
        }
        "must be a character(1), an nn_module_generator, or a function returning an nn_module"
      })

      private$.param_set_base = ps(
        arch_type = p_fct(levels = c("tabm", "tabm-mini", "tabm-packed"),
          init = "tabm", tags = "train"),
        k = p_int(lower = 1L, init = 32L, tags = "train"),
        # no init: the default depends on `num_embeddings` (upstream `TabM.make()`)
        n_blocks = p_int(lower = 1L, tags = "train"),
        d_block = p_int(lower = 1L, init = 512L, tags = "train"),
        dropout = p_dbl(lower = 0, upper = 1, init = 0.1, tags = "train"),
        activation = p_uty(init = "relu", tags = "train", custom_check = check_activation),
        # no init: the default depends on `num_embeddings` (upstream `TabM.make()`)
        start_scaling_init = p_fct(levels = c("random-signs", "normal"), tags = "train"),
        # embeddings for the numerical features
        num_embeddings = p_fct(
          levels = c("none", "linear_relu", "periodic", "piecewise_linear"),
          init = "none", tags = "train"),
        d_embedding = p_int(lower = 1L, tags = "train"),
        n_frequencies = p_int(lower = 1L, init = 48L, tags = "train"),
        frequency_init_scale = p_dbl(lower = 0, init = 0.01, tags = "train"),
        lite = p_lgl(init = FALSE, tags = "train"),
        embedding_activation = p_lgl(tags = "train"),
        n_bins = p_int(lower = 2L, init = 48L, tags = "train")
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
        out$x_cat = ingress_categ(shape = c(NA, n_categ))
      }
      out
    },
    .network = function(task, param_vals) {
      arch_type = param_vals$arch_type %??% "tabm"
      n_num = n_num_features(task)

      # the bins of the piecewise-linear embeddings must be computed on the training data
      x_num = if (identical(param_vals$num_embeddings, "piecewise_linear") && n_num > 0L) {
        # the ingress token defines the column order the network will see, and it is not
        # always the order of `task$feature_names` (e.g. after `po("scale")`)
        num_features = ingress_num()$features(task)
        batchgetter_num(task$data(cols = num_features))
      }
      num_embeddings = tabm_make_num_embeddings(param_vals$num_embeddings, n_num, param_vals, x_num)

      nn_tabm(
        n_num_features = n_num,
        cat_cardinalities = unname(categ_cardinalities(task)),
        d_out = output_dim_for(task),
        num_embeddings = num_embeddings,
        arch_type = arch_type,
        k = param_vals$k %??% 32L,
        # NULL lets nn_tabm() apply upstream's `TabM.make()` defaults, which depend on
        # whether embeddings are used
        n_blocks = param_vals$n_blocks,
        d_block = param_vals$d_block %??% 512L,
        dropout = param_vals$dropout %??% 0.1,
        activation = param_vals$activation %??% "relu",
        # upstream requires this to be NULL for the packed architecture
        start_scaling_init = if (arch_type == "tabm-packed") NULL else param_vals$start_scaling_init
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
