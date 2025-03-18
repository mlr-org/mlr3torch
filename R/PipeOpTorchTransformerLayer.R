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
        ffn_activation = p_uty(),
        residual_dropout = p_dbl(lower = 0, upper = 1, default = 0.0),
        prenormalization = p_lgl(default = TRUE), 
        first_layer = p_lgl(default = FALSE),
        last_layer_query_idx = p_uty(default = NULL, custom_check = function(input) check_integer(input, null.ok = TRUE))
      )
      
      super$initialize(
        id = id,
        module_generator = nn_transformer_layer,
        param_vals = param_vals,
        param_set = param_set
      )
    }
  ),
  private = list(
    .shapes_out = function(shapes_in, param_vals, task) {
      if (is.null(param_vals$last_layer_query_idx)) {
        return(shapes_in)
      }
      
      shapes_out = shapes_in
      shapes_out[[2]] = length(param_vals$last_layer_query_idx)
      return(shapes_out)
    }
  )
)
mlr3pipelines::mlr_pipeops$add("transformer_layer", PipeOpTorchTransformerLayer)

nn_transformer_layer = nn_module(
  "nn_transformer_layer",
  initialize = function(d_token,
                        attention_n_heads,
                        attention_dropout,
                        attention_initialization,
                        ffn_d_hidden,
                        ffn_dropout,
                        ffn_activation,
                        residual_dropout,
                        prenormalization,
                        first_layer = FALSE,
                        attention_normalization,
                        ffn_normalization,
                        kv_compression_ratio = NULL,
                        kv_compression_sharing = NULL,
                        query_idx) {
    self$prenormalization = prenormalization
    
    self$attention = nn_ft_multi_head_attention(
      d_token = d_token,
      n_heads = attention_n_heads,
      dropout = attention_dropout,
      bias = TRUE,
      initialization = attention_initialization
    )
    
    self$ffn = nn_ft_ffn(
      d_token = d_token,
      d_hidden = ffn_d_hidden,
      bias_first = TRUE,
      bias_second = TRUE,
      dropout = ffn_dropout,
      activation = ffn_activation
    )

    self$attention_residual_dropout = nn_dropout(residual_dropout)
    self$ffn_residual_dropout = nn_dropout(residual_dropout)

    self$output = nn_identity()
        
    # TODO: remove layer_idx and ask about how we want to handle this condition
    layer_idx = -1
    if (layer_idx || !prenormalization || first_prenormalization) {
      self$attention_normalization = attention_normalization(d_token)
    }
    self$ffn_normalization = ffn_normalization(d_token)
    if (!is.null(kv_compression_ratio) && is.null(self$shared_kv_compression)) {
      self$key_compression = make_kv_compression(n_tokens, kv_compression_ratio)
      if (kv_compression_sharing == 'headwise') {
        self$value_compression = make_kv_compression(n_tokens, kv_compression_ratio)
      } else {
        assert_true(kv_compression_sharing == 'key_value', "kv_compression_sharing parameter should be set to either 'headwise' or 'key_value'!")
      }
    }
    self$query_idx = query_idx
  },
  start_residual_ = function(stage, x) {
    x_residual = x
    if (self$prenormalization) {
      norm_key = paste0(stage, "_normalization")
      if (norm_key %in% names(self)) {
        x_residual = self[[norm_key]](x_residual)
      }
    }
    return(x_residual)
  },

  make_kv_compression = function(n_tokens, kv_compression_ratio) {
    assert_true(n_tokens && kv_compression_ratio)
    return(nn_linear(n_tokens, floor(n_tokens * kv_compression_ratio), bias=FALSE))
  },
  get_kv_compressions_ = function() {
    if(!is.null(self$shared_kv_compression)) {
      result = c(self$shared_kv_compression, self$shared_kv_compression)
    } else {
      if ("key_compression" %in% names(self) && "value_compression" %in% names(self)) {
        result = c(self$key_compression, self$value_compression)
      } else {
        if ("key_compression" %in% names(self)) {
          result = c(self$key_compression, self$key_compression)
        } else {
          result = NULL
        }
      }
    }
    return(result)
  },
  end_residual_ = function(stage, x, x_residual) {
    x_residual = self[[paste0(stage, "_residual_dropout")]](x_residual)
    x = x + x_residual
    if (!self$prenormalization) {
      x = layer[[paste0(stage, "_normalization")]](x)
    }
    return(x)
  },
  forward = function(x) {
    x_residual = self$start_residual_('attention', x)

    x_residual_arg = if (is.null(self$query_idx)) x_residual else x_residual[, self$query_idx]
    compressions = self$get_kv_compressions_()
    x_residual_vec = self$attention(x_residual_arg,
                                      x_residual,
                                      compressions[1],
                                      compressions[2])
    x = if (!is.null(self$query_idx)) x[, self$query_idx] else x
    x = self$end_residual_('attention', x, x_residual)

    x_residual = self$start_residual_('ffn', x)
    x_residual = self$ffn(x_residual)
    x = self$end_residual_('ffn', x, x_residual)

    x = self$output(x)
    return(x)
  }
)