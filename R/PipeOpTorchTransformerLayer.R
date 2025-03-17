#' @title PipeOpTorchTransformerLayer
#' @description PipeOp for a single transformer layer of the FT-Transformer
PipeOpTorchTransformerLayer = R6::R6Class("PipeOpTorchTransformerLayer",
  inherit = PipeOpTorch,
  public = list(
    #' @description Create a new instance of this [R6][R6::R6Class] class.
    #' @param id (`character(1)`)\cr
    #'   Identifier of the resulting object.
    initialize = function(id = "transformer_layer", param_vals = list()) {
      param_set = ps(
        d_token = p_int(lower = 1L, default = 192L),
        attention_n_heads = p_int(lower = 1L, default = 8L),
        attention_dropout = p_dbl(lower = 0, upper = 1, default = 0.2),
        attention_initialization = p_fct(levels = c("kaiming", "xavier"), default = "kaiming"),
        ffn_d_hidden = p_int(lower = 1L, default = 256L),
        ffn_dropout = p_dbl(lower = 0, upper = 1, default = 0.1),
        ffn_activation = p_uty(default = nn_reglu()),
        residual_dropout = p_dbl(lower = 0, upper = 1, default = 0.0),
        prenormalization = p_lgl(default = TRUE), 
        first_layer = p_lgl(default = FALSE),
        last_layer = p_lgl(default = FALSE)
      )
      
      super$initialize(
        id = id,
        module_generator = function(d_token, attention_n_heads, attention_dropout, 
                                   attention_initialization, ffn_d_hidden, ffn_dropout,
                                   ffn_activation, residual_dropout, prenorm,
                                   first_layer, last_layer) {
          nn_transformer_layer(
            d_token = d_token,
            attention_n_heads = attention_n_heads,
            attention_dropout = attention_dropout,
            attention_initialization = attention_initialization,
            ffn_d_hidden = ffn_d_hidden,
            ffn_dropout = ffn_dropout,
            ffn_activation = ffn_activation,
            residual_dropout = residual_dropout,
            prenormalization = prenorm,
            first_layer = first_layer,
            attention_normalization = nn_layer_norm,
            ffn_normalization = nn_layer_norm,
            kv_compression_ratio = NULL,
            kv_compression_sharing = NULL,
            query_idx = if (last_layer) 1 else NULL # Use 0 for CLS token if last layer
          )
        },
        param_vals = param_vals,
        param_set = param_set
      )
    }
  ),
  private = list(
    .shapes_out = function(shapes_in, param_vals, task) {
      # Transformer layer preserves input shape
      return(shapes_in)
    }
  )
)
mlr3pipelines::mlr_pipeops$add("torch_transformer_layer", PipeOpTorchTransformerLayer)
