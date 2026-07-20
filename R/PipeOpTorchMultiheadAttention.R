#' @title Multi-Head Attention
#'
#' @description
#' Multi-head attention as described in *Attention Is All You Need*.
#'
#' This is a thin wrapper around [`torch::nn_multihead_attention()`] that makes it usable as a
#' building block of a [`Graph`][mlr3pipelines::Graph] of tensor operations:
#' 1. The `forward()` method accepts between one and three tensors, so that both self-attention and
#'    cross-attention can be expressed.
#' 2. The `forward()` method returns a bare tensor when `need_weights` is `FALSE` and a named
#'    `list()` with elements `"output"` and `"weights"` when `need_weights` is `TRUE`.
#'
#' The `forward()` method accepts between one and three tensors:
#' * one tensor `(query)`: self-attention, i.e. the tensor is used as query, key and value.
#' * two tensors `(query, key_value)`: cross-attention, where the second tensor is used as both key
#'   and value.
#' * three tensors `(query, key, value)`: cross-attention with separate key and value tensors.
#'
#' @param embed_dim (`integer(1)`)\cr
#'   Total dimension of the model, i.e. the size of the last dimension of the query tensor.
#' @param num_heads (`integer(1)`)\cr
#'   Number of parallel attention heads. `embed_dim` must be divisible by `num_heads`.
#' @param dropout (`numeric(1)`)\cr
#'   Dropout probability on the attention weights. Default is `0`.
#' @param bias (`logical(1)`)\cr
#'   Whether to add a bias to the input and output projections. Default is `TRUE`.
#' @param add_bias_kv (`logical(1)`)\cr
#'   Whether to add a bias to the key and value sequences at dimension 1. Default is `FALSE`.
#' @param add_zero_attn (`logical(1)`)\cr
#'   Whether to add a new batch of zeros to the key and value sequences at dimension 1.
#'   Default is `FALSE`.
#' @param kdim (`integer(1)`)\cr
#'   Total number of features for the keys. Default is `NULL`, which means `embed_dim`.
#' @param vdim (`integer(1)`)\cr
#'   Total number of features for the values. Default is `NULL`, which means `embed_dim`.
#' @param batch_first (`logical(1)`)\cr
#'   Whether the input and output tensors are provided as `(batch, sequence, feature)` (`TRUE`) or
#'   as `(sequence, batch, feature)` (`FALSE`). Default is `FALSE`, as in `torch`.
#' @param need_weights (`logical(1)`)\cr
#'   Whether the attention weights are returned in addition to the attention output.
#'   Default is `FALSE`.
#' @param avg_weights (`logical(1)`)\cr
#'   Whether the returned attention weights are averaged over the attention heads.
#'   Default is `TRUE`. Only has an effect when `need_weights` is `TRUE`.\cr
#'   Note that [`torch::nn_multihead_attention()`] silently ignores this argument (and behaves as if
#'   it were `TRUE`) whenever `kdim` or `vdim` differ from `embed_dim`.
#'
#' @references
#' `r format_bib("vaswani2017attention")`
#'
#' @export
nn_attention = nn_module(
  "nn_attention",
  initialize = function(embed_dim, num_heads, dropout = 0, bias = TRUE, add_bias_kv = FALSE,
    add_zero_attn = FALSE, kdim = NULL, vdim = NULL, batch_first = FALSE, need_weights = FALSE,
    avg_weights = TRUE) {
    self$need_weights = assert_flag(need_weights)
    self$avg_weights = assert_flag(avg_weights)
    self$attention = torch::nn_multihead_attention(
      embed_dim = embed_dim,
      num_heads = num_heads,
      dropout = dropout,
      bias = bias,
      add_bias_kv = add_bias_kv,
      add_zero_attn = add_zero_attn,
      kdim = kdim,
      vdim = vdim,
      batch_first = batch_first
    )
  },
  forward = function(...) {
    inputs = list(...)
    query = inputs[[1L]]
    key = if (length(inputs) >= 2L) inputs[[2L]] else query
    value = if (length(inputs) >= 3L) inputs[[3L]] else key
    out = self$attention(query = query, key = key, value = value,
      need_weights = self$need_weights, avg_weights = self$avg_weights)
    if (self$need_weights) {
      list(output = out[[1L]], weights = out[[2L]])
    } else {
      out[[1L]]
    }
  }
)

#' @title Multi-Head Attention
#'
#' @inherit nn_attention description
#' @section nn_module:
#' Calls [`nn_attention()`] when trained, where the parameters `embed_dim`, `kdim` and `vdim` are
#' inferred as the last dimension of the query, key and value tensors respectively.
#' @section Parameters:
#' * `num_heads` :: `integer(1)`\cr
#'   Number of parallel attention heads. The embedding dimension must be divisible by `num_heads`.
#' * `dropout` :: `numeric(1)`\cr
#'   Dropout probability on the attention weights.
#'   Default is `0`.
#' * `bias` :: `logical(1)`\cr
#'   Whether to add a bias to the input and output projections.
#'   Default is `TRUE`.
#' * `add_bias_kv` :: `logical(1)`\cr
#'   Whether to add a bias to the key and value sequences at dimension 1.
#'   Default is `FALSE`.
#' * `add_zero_attn` :: `logical(1)`\cr
#'   Whether to add a new batch of zeros to the key and value sequences at dimension 1.
#'   Default is `FALSE`.
#' * `batch_first` :: `logical(1)`\cr
#'   Whether the input and output tensors are provided as `(batch, sequence, feature)` (`TRUE`) or
#'   as `(sequence, batch, feature)` (`FALSE`).
#'   Default is `FALSE`, as in `torch`.
#' * `avg_weights` :: `logical(1)`\cr
#'   Whether the returned attention weights are averaged over the attention heads.
#'   Default is `TRUE`. Only has an effect when the construction argument `outnum` is 2.
#'
#' Note that `embed_dim`, `kdim` and `vdim` are *not* parameters, as they are inferred from the
#' shapes of the input tensors.
#'
#' @section Input and Output Channels:
#' The number of input channels is determined by the construction argument `innum`:
#' * `innum = 1` (default): one input channel `"input"`, which is used as query, key and value,
#'   i.e. the `PipeOp` performs *self-attention*.
#' * `innum = 2`: input channels `"query"` and `"key_value"`, i.e. the `PipeOp` performs
#'   *cross-attention*, where the second input is used as both key and value.
#' * `innum = 3`: input channels `"query"`, `"key"` and `"value"`, i.e. the `PipeOp` performs
#'   *cross-attention* with separate key and value inputs.
#'
#' The number of output channels is determined by the construction argument `outnum`:
#' * `outnum = 1` (default): one output channel `"output"`, containing the attention output.
#' * `outnum = 2`: output channels `"output"` and `"weights"`, where the latter contains the
#'   attention weights.
#'
#' For an explanation see [`PipeOpTorch`].
#'
#' @section Internals:
#' All input tensors must be three-dimensional. Depending on the parameter `batch_first`, they are
#' interpreted as `(batch, sequence, feature)` or as `(sequence, batch, feature)`.
#' The feature dimension is the last dimension in both layouts.
#'
#' The shape of the attention output is identical to the shape of the query tensor (in both
#' layouts). The attention weights, on the other hand, are **always** batch-first, irrespective of
#' `batch_first`, because [`torch::nnf_multi_head_attention_forward()`] transposes its inputs to a
#' sequence-first layout before extracting the batch size. Their shape is
#' `(batch, query_sequence, key_sequence)` if the weights are averaged over the heads and
#' `(batch, num_heads, query_sequence, key_sequence)` otherwise, where `key_sequence` is increased
#' by 1 for each of `add_bias_kv` and `add_zero_attn`.
#'
#' Because [`torch::nn_multihead_attention()`] does not forward `avg_weights` when `kdim` or `vdim`
#' differ from `embed_dim`, the weights are averaged in that case even if `avg_weights` is `FALSE`.
#' The shape inference accounts for this.
#'
#' @templateVar id nn_multihead_attention
#' @templateVar param_vals num_heads = 4
#' @template pipeop_torch
#' @template pipeop_torch_example
#'
#' @export
PipeOpTorchMultiheadAttention = R6Class("PipeOpTorchMultiheadAttention",
  inherit = PipeOpTorch,
  public = list(
    #' @description Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    #' @param innum (`integer(1)`)\cr
    #'   The number of input channels, which must be between 1 and 3.
    #'   This is a *construction* argument (and not a hyperparameter), because it determines the
    #'   structure of the [`Graph`][mlr3pipelines::Graph].
    #'   The default is 1, which means that the `PipeOp` performs self-attention.
    #'   See section *Input and Output Channels* for more information.
    #' @param outnum (`integer(1)`)\cr
    #'   The number of output channels, which must be 1 or 2.
    #'   This is a *construction* argument (and not a hyperparameter), because it determines the
    #'   structure of the [`Graph`][mlr3pipelines::Graph].
    #'   The default is 1, which means that only the attention output is returned.
    #'   See section *Input and Output Channels* for more information.
    initialize = function(id = "nn_multihead_attention", innum = 1, outnum = 1, param_vals = list()) {
      assert_int(innum, lower = 1, upper = 3)
      assert_int(outnum, lower = 1, upper = 2)
      private$.innum = as.integer(innum)
      private$.outnum = as.integer(outnum)
      inname = switch(private$.innum,
        "input",
        c("query", "key_value"),
        c("query", "key", "value")
      )
      outname = switch(private$.outnum,
        "output",
        c("output", "weights")
      )
      param_set = ps(
        num_heads = p_int(lower = 1L, tags = c("train", "required")),
        dropout = p_dbl(lower = 0, upper = 1, default = 0, tags = "train"),
        bias = p_lgl(default = TRUE, tags = "train"),
        add_bias_kv = p_lgl(default = FALSE, tags = "train"),
        add_zero_attn = p_lgl(default = FALSE, tags = "train"),
        batch_first = p_lgl(default = FALSE, tags = "train"),
        avg_weights = p_lgl(default = TRUE, tags = "train")
      )
      super$initialize(
        id = id,
        module_generator = nn_attention,
        param_set = param_set,
        param_vals = param_vals,
        inname = inname,
        outname = outname,
        # the batch dimension is not necessarily the first one, so other dimensions can be unknown
        only_batch_unknown = FALSE
      )
    }
  ),
  private = list(
    .innum = NULL,
    .outnum = NULL,
    .additional_phash_input = function() {
      list(private$.innum, private$.outnum)
    },
    # the shape of the key input, which for `innum == 1` is the query itself
    .key_shape = function(shapes_in) {
      if (private$.innum == 1L) shapes_in[[1L]] else shapes_in[[2L]]
    },
    # whether torch takes the `qkv_same_embed_dim_` branch, which is the only one that forwards
    # `avg_weights` to `nnf_multi_head_attention_forward()`
    .qkv_same_embed_dim = function(shapes_in) {
      embed_dim = tail(shapes_in[[1L]], 1L)
      kdim = tail(private$.key_shape(shapes_in), 1L)
      vdim = tail(shapes_in[[private$.innum]], 1L)
      isTRUE(kdim == embed_dim) && isTRUE(vdim == embed_dim)
    },
    .shapes_out = function(shapes_in, param_vals, task) {
      walk(seq_along(shapes_in), function(i) {
        if (length(shapes_in[[i]]) != 3L) {
          stopf("PipeOpTorchMultiheadAttention expects three-dimensional inputs, but input %i has %i dimensions.", i, length(shapes_in[[i]])) # nolint
        }
      })
      query_shape = shapes_in[[1L]]
      embed_dim = tail(query_shape, 1L)
      if (is.na(embed_dim)) {
        stopf("PipeOpTorchMultiheadAttention received an input shape where the last dimension is unknown. Please provide an input with a known last dimension.") # nolint
      }
      if (embed_dim %% param_vals$num_heads != 0) {
        stopf("PipeOpTorchMultiheadAttention: the embedding dimension (%i) must be divisible by 'num_heads' (%i).", embed_dim, param_vals$num_heads) # nolint
      }
      # the attention output has the same shape as the query, in both layouts
      if (private$.outnum == 1L) {
        return(list(query_shape))
      }

      batch_first = param_vals$batch_first %??% FALSE
      key_shape = private$.key_shape(shapes_in)
      # the attention weights are always batch-first, irrespective of `batch_first`
      n_batch = if (batch_first) query_shape[1L] else query_shape[2L]
      tgt_len = if (batch_first) query_shape[2L] else query_shape[1L]
      src_len = if (batch_first) key_shape[2L] else key_shape[1L]
      src_len = src_len + (param_vals$add_bias_kv %??% FALSE) + (param_vals$add_zero_attn %??% FALSE)

      # torch ignores `avg_weights` unless kdim and vdim both equal embed_dim
      avg_weights = (param_vals$avg_weights %??% TRUE) || !private$.qkv_same_embed_dim(shapes_in)
      weights_shape = if (avg_weights) {
        c(n_batch, tgt_len, src_len)
      } else {
        c(n_batch, param_vals$num_heads, tgt_len, src_len)
      }
      list(query_shape, weights_shape)
    },
    .shape_dependent_params = function(shapes_in, param_vals, task) {
      param_vals$embed_dim = tail(shapes_in[[1L]], 1L)
      if (private$.innum > 1L) {
        # for innum == 2, the second channel provides both the keys and the values
        param_vals$kdim = tail(shapes_in[[2L]], 1L)
        param_vals$vdim = tail(shapes_in[[private$.innum]], 1L)
      }
      param_vals$need_weights = private$.outnum == 2L
      param_vals
    }
  )
)

#' @include aaa.R
register_po("nn_multihead_attention", PipeOpTorchMultiheadAttention)
