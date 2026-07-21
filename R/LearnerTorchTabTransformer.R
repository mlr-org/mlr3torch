# ------------------------------------------------------------------------------------------------
# TabTransformer (Huang et al., 2020, arXiv:2012.06678)
#
# Ported from:
#   LAMDA-Tabular/TALENT -- https://github.com/LAMDA-Tabular/TALENT
#   file:    TALENT/model/models/tabtransformer.py
#   commit:  08301d670a7c854bcf3a73298763484ba58eecdb
#   license: MIT (Copyright (c) 2024 LAMDA-Tabular), see inst/COPYRIGHTS
#
# Cross-checked against:
#   lucidrains/tab-transformer-pytorch -- https://github.com/lucidrains/tab-transformer-pytorch
#   file:    tab_transformer_pytorch/tab_transformer_pytorch.py
#   tag:     0.2.6, commit 742ac84f3a4dcdfc46b18b6d29db28afaf3048a0
#   license: MIT (Copyright (c) 2020 Phil Wang)
# and against the paper itself.
#
# Why not lucidrains' `main` branch: the current upstream `main` has diverged substantially from
# the paper. It now depends on the external packages `x_mlps_pytorch`, `hyper_connections`
# (multi-residual-stream hyper-connections) and `discrete_continuous_embed_readout`, none of which
# appear in the paper. TALENT's copy (whose own header states it is "adapted from
# https://github.com/lucidrains/tab-transformer-pytorch") is self-contained, only needs `einops`,
# and still matches the paper, so it is the reference used here.
#
# Where TALENT and lucidrains@0.2.6 disagree, the paper decides:
#   * Shared column embedding: TALENT has `use_shared_categ_embed` / `shared_categ_dim_divisor`,
#     lucidrains@0.2.6 does not (it was added upstream after that tag). The paper (Section 2,
#     "column embedding") reserves d/8 of the token dimension for a per-column shared embedding,
#     so TALENT's behaviour is followed and is the default.
#   * MLP head widths: TALENT uses `input_size * mlp_hidden_mults`, lucidrains@0.2.6 uses
#     `(input_size %/% 8) * mlp_hidden_mults`. The paper (Appendix, "MLP") specifies hidden layer
#     sizes {4*l, 2*l} where l is the size of the MLP input, i.e. TALENT's version. TALENT is
#     followed.
#
# Intentional deviations from upstream:
#   * `num_special_tokens` (upstream default 2) is dropped, i.e. effectively fixed to 0. The two
#     reserved embedding rows are never indexed by any input mlr3torch can produce (factor levels
#     are always complete and missing values are not representable), so they would be dead
#     parameters that never receive a gradient. Numerically this changes nothing.
#   * Categorical codes: `batchgetter_categ()` produces 1-based codes for `factor`/`ordered` and
#     0-based codes (0/1) for `logical`. The cumulative offsets are therefore shifted by +1 for
#     logical columns so that every column still indexes a contiguous block of the (1-based) R
#     embedding table. Upstream assumes 0-based codes throughout.
#   * `continuous_mean_std` is not exposed. Feature standardisation is the job of an mlr3 pipeline
#     (e.g. `po("scale")`), not of the network.
#   * `mlp_act` is not exposed; the head always uses ReLU (upstream's default).
#   * Upstream always builds the transformer stack, even when there are no categorical features
#     (`categories = None`), leaving it with dead parameters. Here the categorical branch
#     (embeddings + transformer) is only built when the task has categorical features, and the
#     `nn_layer_norm` over the continuous features is only built when the task has numeric
#     features. This keeps every parameter reachable by gradients.
#   * Upstream returns attention maps via `return_attn`; that is dropped, the module returns the
#     logits only.
#   * Upstream's unused `Residual` helper class is not ported (upstream inlines the residual
#     connections in `Transformer.forward()` and never instantiates `Residual`).
# ------------------------------------------------------------------------------------------------

#' @title Attention Block of a TabTransformer
#' @name nn_tab_transformer_attention
#' @description
#' Multi-head self-attention as used by [`nn_tab_transformer()`].
#' @noRd
nn_tab_transformer_attention = nn_module("nn_tab_transformer_attention",
  initialize = function(d_token, n_heads = 8L, dim_head = 16L, dropout = 0) {
    self$n_heads = assert_int(n_heads, lower = 1L, coerce = TRUE)
    self$dim_head = assert_int(dim_head, lower = 1L, coerce = TRUE)
    inner_dim = self$n_heads * self$dim_head
    self$scale = self$dim_head^(-0.5)
    self$to_qkv = nn_linear(d_token, inner_dim * 3L, bias = FALSE)
    self$to_out = nn_linear(inner_dim, d_token)
    self$dropout = nn_dropout(dropout)
  },
  forward = function(input) {
    # `n` (the number of tokens) is constant across batches, the batch size is not, hence the -1
    # (this keeps the module traceable with `jit_trace()`)
    n = input$shape[2L]
    qkv = self$to_qkv(input)$chunk(3L, dim = -1L)
    # einops: 'b n (h d) -> b h n d'
    split_heads = function(x) x$view(c(-1L, n, self$n_heads, self$dim_head))$transpose(2L, 3L)
    q = split_heads(qkv[[1L]])
    k = split_heads(qkv[[2L]])
    v = split_heads(qkv[[3L]])

    sim = torch_matmul(q, k$transpose(-2L, -1L)) * self$scale
    attn = nnf_softmax(sim, dim = -1L)
    attn = self$dropout(attn)

    out = torch_matmul(attn, v)
    # einops: 'b h n d -> b n (h d)'
    out = out$transpose(2L, 3L)$reshape(c(-1L, n, self$n_heads * self$dim_head))
    self$to_out(out)
  }
)

#' @title GEGLU Feed-Forward Block of a TabTransformer
#' @name nn_tab_transformer_ffn
#' @description
#' Position-wise feed-forward network with a GEGLU activation, as used by
#' [`nn_tab_transformer()`].
#' @noRd
nn_tab_transformer_ffn = nn_module("nn_tab_transformer_ffn",
  initialize = function(d_token, mult = 4L, dropout = 0) {
    d_hidden = d_token * mult
    self$linear_first = nn_linear(d_token, d_hidden * 2L)
    self$dropout = nn_dropout(dropout)
    self$linear_second = nn_linear(d_hidden, d_token)
  },
  forward = function(input) {
    x = self$linear_first(input)
    chunks = x$chunk(2L, dim = -1L)
    x = chunks[[1L]] * nnf_gelu(chunks[[2L]])
    x = self$dropout(x)
    self$linear_second(x)
  }
)

#' @title Transformer Block of a TabTransformer
#' @name nn_tab_transformer_block
#' @description
#' Pre-normalized attention and feed-forward sub-layers with residual connections.
#' @noRd
nn_tab_transformer_block = nn_module("nn_tab_transformer_block",
  initialize = function(d_token, n_heads, dim_head, attn_dropout, ff_dropout) {
    self$attention_norm = nn_layer_norm(d_token)
    self$attention = nn_tab_transformer_attention(
      d_token = d_token, n_heads = n_heads, dim_head = dim_head, dropout = attn_dropout
    )
    self$ffn_norm = nn_layer_norm(d_token)
    self$ffn = nn_tab_transformer_ffn(d_token = d_token, dropout = ff_dropout)
  },
  forward = function(input) {
    x = input + self$attention(self$attention_norm(input))
    x + self$ffn(self$ffn_norm(x))
  }
)

#' @title Cardinalities of the Categorical Features of a Task
#' @description
#' Returns the number of levels of the categorical features of a task, in the same order in which
#' [`ingress_categ()`] selects them. `logical` features have no levels in `mlr3` and are counted
#' as having cardinality 2.
#' @noRd
categ_cardinalities = function(task) {
  features = ingress_categ()$features(task)
  types = categ_types(task)
  levels = task$levels(features)
  set_names(ifelse(types == "logical", 2L, lengths(levels[features])), features)
}

#' @title Feature Types of the Categorical Features of a Task
#' @description
#' In the same order in which [`ingress_categ()`] selects them.
#' @noRd
categ_types = function(task) {
  features = ingress_categ()$features(task)
  ft = task$feature_types
  ft$type[match(features, ft$id)]
}

#' @title TabTransformer Network
#' @name nn_tab_transformer
#' @description
#' Implements the TabTransformer architecture of Huang et al. (2020).
#'
#' The categorical features are embedded into a `d_token`-dimensional space (optionally reserving
#' `d_token %/% shared_categ_dim_divisor` dimensions for a per-column *shared* embedding) and are
#' then contextualized by a stack of `depth` transformer blocks. The resulting contextual
#' embeddings of shape `(batch, n_categorical, d_token)` are flattened and concatenated with the
#' layer-normalized numeric features. An MLP with hidden widths
#' `mlp_hidden_mults * (d_token * n_categorical + n_numeric)` maps this to the output.
#'
#' Only the categorical features pass through the transformer; the numeric features bypass it.
#'
#' @param task ([`Task`][mlr3::Task] or `NULL`)\cr
#'   The task for which to construct the network. If provided, `cardinalities`, `n_features_num`
#'   and `d_out` are inferred from it and must not be given.
#' @param cardinalities (`integer()`)\cr
#'   The number of levels of each categorical feature. Must be in the same order as the columns of
#'   `x_cat`. Only needed if `task` is `NULL`.
#' @param n_features_num (`integer(1)`)\cr
#'   The number of numeric features, i.e. the number of columns of `x_num`.
#'   Only needed if `task` is `NULL`.
#' @param d_out (`integer(1)`)\cr
#'   The output dimension of the network. Only needed if `task` is `NULL`.
#' @param d_token (`integer(1)`)\cr
#'   The dimension of the categorical embeddings, i.e. the width of the transformer.
#' @param depth (`integer(1)`)\cr
#'   The number of transformer blocks.
#' @param n_heads (`integer(1)`)\cr
#'   The number of attention heads.
#' @param dim_head (`integer(1)`)\cr
#'   The dimension of each attention head.
#' @param attn_dropout (`numeric(1)`)\cr
#'   The dropout probability applied to the attention weights.
#' @param ff_dropout (`numeric(1)`)\cr
#'   The dropout probability applied inside the feed-forward blocks.
#' @param mlp_hidden_mults (`integer()`)\cr
#'   Multipliers defining the hidden widths of the output MLP, relative to the size of its input.
#' @param use_shared_categ_embed (`logical(1)`)\cr
#'   Whether to reserve part of the embedding dimension for a per-column embedding that is shared
#'   by all levels of that column.
#' @param shared_categ_dim_divisor (`numeric(1)`)\cr
#'   The shared column embedding uses `d_token %/% shared_categ_dim_divisor` dimensions.
#'   Must be at least `2`, so that at most half of the token is the shared embedding.
#'
#' @section Input and Output:
#' The `forward()` method takes the arguments `x_num` (a `float` tensor of shape
#' `(batch, n_features_num)`) and `x_cat` (a `long` tensor of shape
#' `(batch, length(cardinalities))`), both of which default to `NULL`.
#' Categorical codes are expected to be 1-based, as produced by [`batchgetter_categ()`].
#' The output has shape `(batch, d_out)`.
#'
#' @references
#' `r format_bib("huang2020tabtransformer", "shazeer2020glu")`
#' @export
#' @examplesIf torch::torch_is_installed()
#' network = nn_tab_transformer(
#'   cardinalities = c(3, 4), n_features_num = 5, d_out = 2,
#'   d_token = 8, depth = 1, n_heads = 2
#' )
#' x_num = torch::torch_randn(2, 5)
#' x_cat = torch::torch_cat(list(
#'   torch::torch_randint(1, 3, c(2, 1), dtype = torch::torch_long()),
#'   torch::torch_randint(1, 4, c(2, 1), dtype = torch::torch_long())
#' ), dim = 2)
#' network(x_num = x_num, x_cat = x_cat)
nn_tab_transformer = nn_module("nn_tab_transformer",
  initialize = function(task = NULL, cardinalities = NULL, n_features_num = NULL, d_out = NULL,
    d_token = 32L, depth = 6L, n_heads = 8L, dim_head = 16L, attn_dropout = 0, ff_dropout = 0,
    mlp_hidden_mults = c(4L, 2L), use_shared_categ_embed = TRUE, shared_categ_dim_divisor = 8) {
    categ_is_logical = NULL
    if (!is.null(task)) {
      assert_class(task, "Task")
      if (!all(map_lgl(list(cardinalities, n_features_num, d_out), is.null))) {
        stopf("When 'task' is provided, 'cardinalities', 'n_features_num' and 'd_out' must not be provided.") # nolint
      }
      supported = c("numeric", "integer", "factor", "ordered", "logical")
      types = task$feature_types$type
      if (!all(types %in% supported)) {
        stopf("nn_tab_transformer() only supports numeric, integer, factor, ordered and logical features, but task '%s' also has: %s.", # nolint
          task$id, paste0(unique(setdiff(types, supported)), collapse = ", "))
      }
      cardinalities = categ_cardinalities(task)
      categ_is_logical = categ_types(task) == "logical"
      n_features_num = n_num_features(task)
      d_out = output_dim_for(task)
    }
    cardinalities = assert_integerish(cardinalities, lower = 1L, any.missing = FALSE,
      null.ok = TRUE, coerce = TRUE)
    n_features_num = assert_int(n_features_num, lower = 0L, coerce = TRUE)
    d_out = assert_int(d_out, lower = 1L, coerce = TRUE)
    d_token = assert_int(d_token, lower = 1L, coerce = TRUE)
    depth = assert_int(depth, lower = 0L, coerce = TRUE)
    n_heads = assert_int(n_heads, lower = 1L, coerce = TRUE)
    dim_head = assert_int(dim_head, lower = 1L, coerce = TRUE)
    assert_number(attn_dropout, lower = 0, upper = 1)
    assert_number(ff_dropout, lower = 0, upper = 1)
    mlp_hidden_mults = assert_numeric(mlp_hidden_mults, lower = 0, any.missing = FALSE,
      min.len = 0L)
    assert_flag(use_shared_categ_embed)
    assert_number(shared_categ_dim_divisor, lower = 2)

    n_features_categ = length(cardinalities)
    if (n_features_categ == 0L && n_features_num == 0L) {
      stopf("nn_tab_transformer() needs at least one feature, but got none.")
    }
    self$n_features_categ = n_features_categ
    self$n_features_num = n_features_num
    self$d_token = d_token
    self$depth = depth
    self$use_shared_categ_embed = use_shared_categ_embed && n_features_categ > 0L

    # ---- categorical branch (embeddings + transformer) -----------------------------------------
    if (n_features_categ > 0L) {
      shared_embed_dim = if (use_shared_categ_embed) d_token %/% shared_categ_dim_divisor else 0L
      if (use_shared_categ_embed && shared_embed_dim < 1L) {
        stopf("nn_tab_transformer(): 'd_token' (%i) must be at least 'shared_categ_dim_divisor' (%s) when 'use_shared_categ_embed' is TRUE.", # nolint
          d_token, format(shared_categ_dim_divisor))
      }
      self$shared_embed_dim = shared_embed_dim
      self$category_embed = nn_embedding(sum(cardinalities), d_token - shared_embed_dim)
      if (use_shared_categ_embed) {
        self$shared_category_embed = nn_parameter(torch_empty(n_features_categ, shared_embed_dim))
      }
      # `batchgetter_categ()` yields 1-based codes for factor/ordered but 0-based codes for
      # logical, hence the additional shift for logical columns.
      if (is.null(categ_is_logical)) {
        categ_is_logical = rep(FALSE, n_features_categ)
      }
      offsets = cumsum(c(0L, cardinalities[-n_features_categ])) + categ_is_logical
      self$category_offsets = nn_buffer(torch_tensor(as.integer(offsets), dtype = torch_long()))
      self$transformer = nn_module_list(map(seq_len(depth), function(i) {
        nn_tab_transformer_block(d_token = d_token, n_heads = n_heads, dim_head = dim_head,
          attn_dropout = attn_dropout, ff_dropout = ff_dropout)
      }))
    }

    # ---- continuous branch ---------------------------------------------------------------------
    if (n_features_num > 0L) {
      self$norm = nn_layer_norm(n_features_num)
    }

    # ---- head ----------------------------------------------------------------------------------
    input_size = d_token * n_features_categ + n_features_num
    dims = c(input_size, as.integer(input_size * mlp_hidden_mults), d_out)
    layers = list()
    for (i in seq_len(length(dims) - 1L)) {
      layers[[length(layers) + 1L]] = nn_linear(dims[[i]], dims[[i + 1L]])
      if (i < length(dims) - 1L) {
        layers[[length(layers) + 1L]] = nn_relu()
      }
    }
    self$mlp = invoke(nn_sequential, .args = layers)

    self$reset_parameters()
  },
  reset_parameters = function() {
    if (self$use_shared_categ_embed) {
      with_no_grad(nn_init_normal_(self$shared_category_embed, std = 0.02))
    }
  },
  forward = function(x_num = NULL, x_cat = NULL) {
    xs = list()
    if (self$n_features_categ > 0L) {
      if (is.null(x_cat)) {
        stopf("nn_tab_transformer(): argument 'x_cat' is required, but was not provided.")
      }
      x = self$category_embed(x_cat + self$category_offsets[NULL])
      if (self$use_shared_categ_embed) {
        # `$expand_as()` on a slice of `x` (rather than `$expand()` with an explicit batch size)
        # keeps the batch dimension dynamic, so the module stays traceable with `jit_trace()`
        shared = self$shared_category_embed$unsqueeze(1L)$expand_as(
          x$narrow(3L, 1L, self$shared_embed_dim))
        x = torch_cat(list(x, shared), dim = -1L)
      }
      for (i in seq_len(self$depth)) {
        x = self$transformer[[i]](x)
      }
      xs[[length(xs) + 1L]] = x$flatten(start_dim = 2L)
    }
    if (self$n_features_num > 0L) {
      if (is.null(x_num)) {
        stopf("nn_tab_transformer(): argument 'x_num' is required, but was not provided.")
      }
      xs[[length(xs) + 1L]] = self$norm(x_num)
    }
    x = if (length(xs) == 1L) xs[[1L]] else torch_cat(xs, dim = -1L)
    self$mlp(x)
  }
)

# Adapts a network whose forward() takes (x_num, x_cat) to a task that has *only* categorical
# features: in that case mlr3torch's training loop passes the single batch tensor by position
# (see `learner_torch_train()`), which would bind it to `x_num`.
nn_tab_transformer_categ_only = nn_module("nn_tab_transformer_categ_only",
  initialize = function(network) {
    self$network = network
  },
  forward = function(x_cat) {
    self$network(x_cat = x_cat)
  }
)

#' @title TabTransformer
#'
#' @templateVar name tab_transformer
#' @templateVar task_types classif, regr
#' @templateVar param_vals d_token = 16, depth = 2, n_heads = 4, mlp_hidden_mults = c(2, 1)
#' @template params_learner
#' @templateVar task_id german_credit
#' @template learner
#' @template learner_example
#'
#' @description
#' TabTransformer for tabular data, as described in Huang et al. (2020).
#'
#' @details
#' The categorical features are embedded and contextualized by a stack of transformer blocks, while
#' the numeric features only pass through a [`nn_layer_norm`][torch::nn_layer_norm] and bypass the
#' transformer entirely. This is the defining difference to the
#' [FT-Transformer][mlr_learners.ft_transformer], which tokenizes the numeric features as
#' well and lets them attend to the categorical ones.
#'
#' The contextualized categorical embeddings are flattened, concatenated with the normalized
#' numeric features and fed into an MLP head.
#'
#' Following the paper, `d_token %/% shared_categ_dim_divisor` dimensions of each embedding are
#' reserved for a per-column embedding that is shared by all levels of that column
#' (`use_shared_categ_embed`).
#'
#' For a task **without categorical features** the transformer has nothing to contextualize and the
#' learner degenerates to an MLP applied to the layer-normalized numeric features. This is
#' supported (and matches the behaviour of the reference implementation for `categories = None`),
#' but in that situation a plain [MLP][mlr_learners.mlp] or
#' [tabular ResNet][mlr_learners.tab_resnet] is the more honest choice. The transformer
#' parameters are not created at all in this case. A task without numeric features is supported as
#' well and is the setting the architecture was designed for.
#'
#' Note that the size of the MLP head grows quadratically in `d_token * n_categorical`, because its
#' hidden widths are defined relative to its own input size (as specified in the paper). For tasks
#' with many categorical features, consider reducing `d_token` or `mlp_hidden_mults`.
#'
#' The default `attn_dropout` and `ff_dropout` of `0` are those of the reference implementation;
#' the TALENT benchmark suite uses `0.08` and `0.3` as its untuned defaults, which is a reasonable
#' starting point.
#'
#' If training is unstable, consider standardizing the numeric features (e.g. using `po("scale")`),
#' reducing the learning rate, and using a learning rate scheduler (see
#' [`CallbackSetLRScheduler`] for options).
#'
#' @section Parameters:
#' Parameters from [`LearnerTorch`], as well as:
#' * `d_token` :: `integer(1)`\cr
#'   The dimension of the categorical embeddings, i.e. the width of the transformer.
#'   Default is `32`.
#' * `depth` :: `integer(1)`\cr
#'   The number of transformer blocks. Default is `6`.
#' * `n_heads` :: `integer(1)`\cr
#'   The number of attention heads. Default is `8`.
#' * `dim_head` :: `integer(1)`\cr
#'   The dimension of each attention head. Default is `16`.
#' * `attn_dropout` :: `numeric(1)`\cr
#'   The dropout probability for the attention weights. Default is `0`.
#' * `ff_dropout` :: `numeric(1)`\cr
#'   The dropout probability in the feed-forward blocks. Default is `0`.
#' * `mlp_hidden_mults` :: `integer()`\cr
#'   Multipliers that define the hidden widths of the output MLP, relative to the size of its
#'   input. Default is `c(4, 2)`.
#' * `use_shared_categ_embed` :: `logical(1)`\cr
#'   Whether to use a shared per-column embedding. Default is `TRUE`.
#' * `shared_categ_dim_divisor` :: `numeric(1)`\cr
#'   The shared column embedding uses `d_token %/% shared_categ_dim_divisor` dimensions.
#'   Must be at least `2`. Default is `8`.
#'
#' @seealso [`nn_tab_transformer()`], [FT-Transformer][mlr_learners.ft_transformer]
#'
#' @references
#' `r format_bib("huang2020tabtransformer")`
#' @export
LearnerTorchTabTransformer = R6Class("LearnerTorchTabTransformer",
  inherit = LearnerTorch,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(task_type, optimizer = NULL, loss = NULL, callbacks = list()) {
      check_mults = crate(function(x) {
        check_numeric(x, lower = 0, any.missing = FALSE, min.len = 0L)
      })
      private$.param_set_base = ps(
        d_token = p_int(lower = 1L, default = 32L, tags = "train"),
        depth = p_int(lower = 0L, default = 6L, tags = "train"),
        n_heads = p_int(lower = 1L, default = 8L, tags = "train"),
        dim_head = p_int(lower = 1L, default = 16L, tags = "train"),
        attn_dropout = p_dbl(lower = 0, upper = 1, default = 0, tags = "train"),
        ff_dropout = p_dbl(lower = 0, upper = 1, default = 0, tags = "train"),
        mlp_hidden_mults = p_uty(default = c(4L, 2L), tags = "train", custom_check = check_mults),
        use_shared_categ_embed = p_lgl(default = TRUE, tags = "train"),
        shared_categ_dim_divisor = p_dbl(lower = 2, default = 8, tags = "train")
      )

      super$initialize(
        task_type = task_type,
        id = paste0(task_type, ".tab_transformer"),
        label = "TabTransformer",
        param_set = alist(private$.param_set_base),
        optimizer = optimizer,
        callbacks = callbacks,
        loss = loss,
        man = "mlr3torch::mlr_learners.tab_transformer",
        feature_types = c("numeric", "integer", "logical", "factor", "ordered"),
        jittable = TRUE
      )
    }
  ),
  private = list(
    .ingress_tokens = function(task, param_vals) {
      n_num = n_num_features(task)
      n_categ = n_categ_features(task)
      if (n_num + n_categ == 0L) {
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
      args = param_vals[intersect(names(param_vals), formalArgs(nn_tab_transformer))]
      network = invoke(nn_tab_transformer, task = task, .args = args)
      if (n_num_features(task) == 0L) {
        # the single input tensor would otherwise be bound to `x_num` positionally
        return(nn_tab_transformer_categ_only(network))
      }
      network
    }
  )
)

#' @include aaa.R
register_learner("classif.tab_transformer", LearnerTorchTabTransformer)
register_learner("regr.tab_transformer", LearnerTorchTabTransformer)
