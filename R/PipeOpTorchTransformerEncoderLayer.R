nn_encoder_layer = nn_module(
  "nn_encoder_layer",
  initialize = function(d_model, nhead, dim_feedforward = 2048, dropout = 0.1, activation = "relu",
    layer_norm_eps = 1e-5, batch_first = FALSE, norm_first = FALSE, bias = TRUE, is_causal = FALSE,
    mask_inputs = character(0)) {
    self$is_causal = assert_flag(is_causal)
    # which optional forward arguments arrive as inputs, in the order of the input channels
    self$mask_inputs = assert_subset(mask_inputs, c("src_mask", "src_key_padding_mask"))
    self$layer = torch::nn_transformer_encoder_layer(
      d_model = d_model,
      nhead = nhead,
      dim_feedforward = dim_feedforward,
      dropout = dropout,
      activation = activation,
      layer_norm_eps = layer_norm_eps,
      batch_first = batch_first,
      norm_first = norm_first,
      bias = bias
    )
  },
  forward = function(...) {
    inputs = list(...)
    args = list(src = inputs[[1L]])
    for (i in seq_along(self$mask_inputs)) {
      args[[self$mask_inputs[[i]]]] = inputs[[i + 1L]]
    }
    if (self$is_causal) {
      # torch's own `is_causal` builds the mask with torch_ones() on the default device, which
      # fails once the network is on the GPU, so the equivalent mask is built here instead: the
      # layout is batch-first, so the sequence length is the second dimension
      len = args$src$shape[[2L]]
      args$src_mask = torch_ones(c(len, len), dtype = torch_bool(),
        device = args$src$device)$triu(diagonal = 1)
    }
    do.call(self$layer, args)
  }
)

#' @title Transformer Encoder Layer
#'
#' @description
#' A single transformer encoder layer as described in *Attention Is All You Need*, consisting of a
#' multi-head self-attention block and a position-wise feed-forward network, each wrapped in a
#' residual connection and layer normalization.
#'
#' This is a thin wrapper around [`torch::nn_transformer_encoder_layer()`] that makes it usable as a
#' building block of a [`Graph`][mlr3pipelines::Graph] of tensor operations, where the attention
#' masks can be supplied as additional inputs, see section *Input and Output Channels*.
#' To build a stack of encoder layers, chain multiple of these `PipeOp`s, giving each its own `id`.
#'
#' @section Tensor Layout:
#' Input and output are `(batch, sequence, feature)`, i.e. the `batch_first` layout of
#' [`torch::nn_transformer_encoder_layer()`], which is fixed and not a hyperparameter.
#' `torch` defaults to `(sequence, batch, feature)`, but the first dimension of every shape has to
#' be the batch dimension here.
#'
#' The layer preserves the shape of its input, so the output shape is the input shape.
#' Only the feature dimension has to be known, the sequence length may be `NA`.
#'
#' @section nn_module:
#' Calls [`torch::nn_transformer_encoder_layer()`] when trained, where the parameter `d_model` is
#' inferred as the last dimension of the input tensor and `batch_first` is always `TRUE`, see
#' section *Tensor Layout*.
#'
#' @section Parameters:
#' * `nhead` :: `integer(1)`\cr
#'   Number of parallel attention heads. The feature dimension must be divisible by `nhead`.
#' * `dim_feedforward` :: `integer(1)`\cr
#'   The hidden dimension of the feed-forward network.
#'   Default is `2048`.
#' * `dropout` :: `numeric(1)`\cr
#'   Dropout probability, applied to both the attention weights and the feed-forward network.
#'   Default is `0.1`.
#' * `activation` :: `character(1)` | `function`\cr
#'   The activation function of the feed-forward network, either `"relu"`, `"gelu"`, or a function
#'   mapping a [`torch_tensor`][torch::torch_tensor] to a [`torch_tensor`][torch::torch_tensor].
#'   Default is `"relu"`.
#' * `layer_norm_eps` :: `numeric(1)`\cr
#'   A value added to the denominator of the layer normalizations for numerical stability.
#'   Default is `1e-5`.
#' * `norm_first` :: `logical(1)`\cr
#'   Whether to apply layer normalization before the attention and feed-forward sublayers
#'   (pre-norm) instead of after them (post-norm).
#'   Default is `FALSE`.
#' * `bias` :: `logical(1)`\cr
#'   Whether the linear layers use a bias and the layer normalizations learn affine parameters.
#'   Default is `TRUE`.
#' * `is_causal` :: `logical(1)`\cr
#'   Whether to apply a causal mask, i.e. whether a position may only attend to the positions up to
#'   and including itself. This cannot be combined with the `"src_mask"` input channel.
#'   Default is `FALSE`.
#'
#' Note that `d_model` is *not* a parameter, as it is inferred from the shape of the input tensor,
#' and that `batch_first` is *not* a parameter either, as it is fixed to `TRUE`, see section
#' *Tensor Layout*.
#'
#' @section Input and Output Channels:
#' There is always one input channel `"input"`, which is the sequence the layer attends over, and
#' one output channel `"output"` of the same shape.
#'
#' The construction arguments `src_mask` and `src_key_padding_mask` each add a further input
#' channel of that name, which is passed to the corresponding argument of the module's `forward()`
#' method. They are construction arguments and not hyperparameters, because they determine the
#' structure of the [`Graph`][mlr3pipelines::Graph]:
#' * `"src_mask"`: an additive or boolean mask over the sequence, of shape `(sequence, sequence)`
#'   or `(batch * nhead, sequence, sequence)`. Positions that are `TRUE` are *not* attended to.
#' * `"src_key_padding_mask"`: a mask of shape `(batch, sequence)` marking the padding positions of
#'   each observation, which are `TRUE`.
#'
#' Note that a `(sequence, sequence)` `"src_mask"` has no batch dimension, unlike every other tensor
#' in a network, so its first dimension is the sequence length rather than the batch size.
#'
#' For an explanation see [`PipeOpTorch`].
#'
#' @references
#' `r format_bib("vaswani2017attention")`
#'
#' @templateVar id nn_transformer_encoder_layer
#' @templateVar param_vals nhead = 4
#' @template pipeop_torch
#' @template pipeop_torch_example
#'
#' @export
PipeOpTorchTransformerEncoderLayer = R6Class("PipeOpTorchTransformerEncoderLayer",
  inherit = PipeOpTorch,
  public = list(
    #' @description Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    #' @param src_mask (`logical(1)`)\cr
    #'   Whether the attention mask is provided as an additional input channel `"src_mask"`.
    #'   This is a *construction* argument (and not a hyperparameter), because it determines the
    #'   structure of the [`Graph`][mlr3pipelines::Graph].
    #'   The default is `FALSE`, i.e. the layer attends over the full sequence.
    #'   See section *Input and Output Channels* for more information.
    #' @param src_key_padding_mask (`logical(1)`)\cr
    #'   Whether the padding mask is provided as an additional input channel
    #'   `"src_key_padding_mask"`.
    #'   This is a *construction* argument (and not a hyperparameter), because it determines the
    #'   structure of the [`Graph`][mlr3pipelines::Graph].
    #'   The default is `FALSE`, i.e. no position is treated as padding.
    #'   See section *Input and Output Channels* for more information.
    initialize = function(id = "nn_transformer_encoder_layer", src_mask = FALSE,
      src_key_padding_mask = FALSE, param_vals = list()) {
      private$.src_mask = assert_flag(src_mask)
      private$.src_key_padding_mask = assert_flag(src_key_padding_mask)
      check_activation = crate(function(x) {
        if (is.function(x)) return(TRUE)
        check_choice(x, c("relu", "gelu"))
      })
      param_set = ps(
        nhead           = p_int(lower = 1L, tags = c("train", "required")),
        dim_feedforward = p_int(lower = 1L, default = 2048L, tags = "train"),
        dropout         = p_dbl(lower = 0, upper = 1, default = 0.1, tags = "train"),
        activation      = p_uty(default = "relu", custom_check = check_activation, tags = "train"),
        layer_norm_eps  = p_dbl(lower = 0, default = 1e-5, tags = "train"),
        norm_first      = p_lgl(default = FALSE, tags = "train"),
        bias            = p_lgl(default = TRUE, tags = "train"),
        is_causal       = p_lgl(default = FALSE, tags = "train")
      )
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals,
        module_generator = nn_encoder_layer,
        inname = c("input", private$.mask_channels())
      )
    }
  ),
  private = list(
    .src_mask = NULL,
    .src_key_padding_mask = NULL,
    # the optional input channels, in the order in which they are declared
    .mask_channels = function() {
      c(if (private$.src_mask) "src_mask", if (private$.src_key_padding_mask) "src_key_padding_mask")
    },
    .additional_phash_input = function() {
      list(private$.src_mask, private$.src_key_padding_mask)
    },
    .shapes_out = function(shapes_in, param_vals, task) {
      shape = shapes_in[[1L]]
      assert_ndim(shape, 3L, id = self$id)
      # the last dimension becomes d_model, which the module needs to size its weights, whereas the
      # sequence length is only needed at runtime and may stay unknown
      assert_known_dims(shape, 3L, "the last dimension (the embedding dimension 'd_model')", self$id)
      d_model = shape[[3L]]
      if (d_model %% param_vals$nhead != 0) {
        stopf("PipeOp '%s': the embedding dimension (%i) must be divisible by 'nhead' (%i).",
          self$id, d_model, param_vals$nhead)
      }
      if (isTRUE(param_vals$is_causal) && private$.src_mask) {
        stopf("PipeOp '%s': 'is_causal' cannot be combined with the 'src_mask' input channel, as they both set the attention mask. Use one or the other.", self$id) # nolint
      }
      # the masks say which positions are attended to, so they leave the shape alone; only their own
      # number of dimensions is checked, as the sizes may legitimately be unknown
      walk(private$.mask_channels(), function(channel) {
        # `$shapes_out()` names the shapes after the input channels
        mask_shape = shapes_in[[channel]]
        ndim = if (channel == "src_mask") c(2L, 3L) else 2L
        if (length(mask_shape) %nin% ndim) {
          stopf("PipeOp '%s': the input channel '%s' expects a shape with %s dimensions, but got %s, which has %i.", # nolint
            self$id, channel, paste0(ndim, collapse = " or "), shape_to_str(mask_shape), length(mask_shape)) # nolint
        }
      })
      # the layer is shape-preserving
      shapes_in["input"]
    },
    .shape_dependent_params = function(shapes_in, param_vals, task) {
      param_vals$batch_first = TRUE
      param_vals$d_model = shapes_in[[1L]][[3L]]
      param_vals$mask_inputs = private$.mask_channels()
      param_vals
    }
  )
)

#' @include aaa.R
register_po("nn_transformer_encoder_layer", PipeOpTorchTransformerEncoderLayer)
