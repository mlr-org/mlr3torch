#' @title CLS Token for FT-Transformer
#' @description
#' Concatenates a CLS token to the input as the last feature.
#' The input shape is expected to be `(batch, n_features, d_token)` and the output shape is
#' `(batch, n_features + 1, d_token)`.
#'
#' This is used in the [`LearnerTorchFTTransformer`].
#'
#' @param d_token (`integer(1)`)\cr
#'   The dimension of the embedding.
#' @param initialization (`character(1)`)\cr
#'   The initialization method for the embedding weights. Possible values are `"uniform"`
#'   and `"normal"`.
#'
#' @references
#' `r format_bib("devlin2018bert")`
#'
#' @export
nn_ft_cls = nn_module(
  "nn_ft_cls",
  initialize = function(d_token, initialization) {
    self$d_token = d_token
    # an individual CLS token
    self$weight = nn_parameter(torch_empty(d_token))
    self$initialization = initialization
    self$reset_parameters()
  },
  reset_parameters = function() {
    initialize_token_(self$weight, d = self$d_token, self$initialization)
  },
  # Repeats the underlying CLS token to create a tensor with the given leading dimensions.
  # Used for creating a batch of CLS tokens
  expand = function(...) {
    leading_dimensions = list(...)
    if (length(leading_dimensions) == 0) {
      return(self$weight)
    }
    new_dims = rep(1, length(leading_dimensions) - 1)
    return(self$weight$view(c(new_dims, -1))$expand(c(leading_dimensions, -1)))
  },
  forward = function(input) {
    return(torch_cat(list(input, self$expand(input$shape[1], 1)), dim = 2))
  }
)

#' @title CLS Token for FT-Transformer
#' @inherit nn_ft_cls description
#' @section nn_module:
#' Calls [`nn_ft_cls()`] when trained.
#' @templateVar id nn_ft_cls
#' @template pipeop_torch
#' @template pipeop_torch_example
#' @export
PipeOpTorchFTCLS = R6::R6Class("PipeOpTorchFTCLS",
inherit = PipeOpTorch,
  public = list(
    #' @description Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    initialize = function(id = "nn_ft_cls", param_vals = list()) {
      param_set = ps(
        initialization = p_fct(tags = c("train"), levels = c("uniform", "normal"), default = "uniform")
      )

      super$initialize(
        id = id,
        module_generator = nn_ft_cls,
        param_vals = param_vals,
        param_set = param_set
      )
    }
  ),
  private = list(
    .shapes_out = function(shapes_in, param_vals, task) {
      if (length(shapes_in$input) != 3) {
        stop("Input tensor must have 3 dimensions.")
      }
      shapes_in[[1]][2] = shapes_in[[1]][2] + 1
      return(shapes_in)
    },
    .shape_dependent_params = function(shapes_in, param_vals, task) {
      param_vals$d_token = shapes_in$input[3]
      return(param_vals)
    }
  )
)

#' @include aaa.R
register_po("nn_ft_cls", PipeOpTorchFTCLS)
