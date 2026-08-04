#' @title Transformer Encoder Layer
#'
#' @description
#' A single transformer encoder layer as described in *Attention Is All You Need*, consisting of a
#' multi-head self-attention block and a position-wise feed-forward network, each wrapped in a
#' residual connection and layer normalization.
#'
#' This is a thin wrapper around [`torch::nn_transformer_encoder_layer()`] that makes it usable as a
#' building block of a [`Graph`][mlr3pipelines::Graph] of tensor operations.
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
#' The `src_mask`, `src_key_padding_mask` and `is_causal` arguments of the module's `forward()`
#' method are not exposed, i.e. the layer attends over the full sequence.
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
#'
#' Note that `d_model` is *not* a parameter, as it is inferred from the shape of the input tensor,
#' and that `batch_first` is *not* a parameter either, as it is fixed to `TRUE`, see section
#' *Tensor Layout*.
#'
#' @references
#' `r format_bib("vaswani2017attention")`
#'
#' @templateVar id nn_transformer_encoder_layer
#' @templateVar param_vals nhead = 4
#' @template pipeop_torch
#' @template pipeop_torch_channels_default
#' @template pipeop_torch_example
#'
#' @export
PipeOpTorchTransformerEncoderLayer = R6Class("PipeOpTorchTransformerEncoderLayer",
  inherit = PipeOpTorch,
  public = list(
    #' @description Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    initialize = function(id = "nn_transformer_encoder_layer", param_vals = list()) {
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
        bias            = p_lgl(default = TRUE, tags = "train")
      )
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals,
        module_generator = nn_transformer_encoder_layer
      )
    }
  ),
  private = list(
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
      # the layer is shape-preserving
      shapes_in
    },
    .shape_dependent_params = function(shapes_in, param_vals, task) {
      param_vals$batch_first = TRUE
      param_vals$d_model = shapes_in[[1L]][[3L]]
      param_vals
    }
  )
)

#' @include aaa.R
register_po("nn_transformer_encoder_layer", PipeOpTorchTransformerEncoderLayer)
