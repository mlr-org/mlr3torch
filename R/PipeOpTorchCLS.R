#' @title PipeOpTorchCLS
#' @description PipeOp that concatenates a CLS token to the input
#' TODO: describe exactly where it is concatenated
PipeOpTorchCLS = R6::R6Class("PipeOpTorchCLS",
  inherit = PipeOpTorch,
  public = list(
    #' @description Create a new instance of this [R6][R6::R6Class] class.
    #' @param id (`character(1)`)\cr
    #'   Identifier of the resulting object.
    initialize = function(id = "cls", param_vals = list()) {
      param_set = ps(
        d_token = p_int(lower = 1L, default = 192L),
        initialization = p_fct(levels = c("uniform", "normal"), default = "uniform")
      )
      
      super$initialize(
        id = id,
        module_generator = nn_cls_token,
        param_vals = param_vals,
        param_set = param_set
      )
    }
  ),
  private = list(
    .shapes_out = function(shapes_in, param_vals, task) {
      # Add 1 to the sequence length dimension (for CLS token)
      shape_out = shapes_in
      shape_out$sequence_length = shape_out$sequence_length + 1
      return(shape_out)
    }
  )
)
mlr3pipelines::mlr_pipeops$add("torch_cls", PipeOpTorchCLS)

initialize_token_ = function(x, d, initialization="") {
  assert_choice(initialization, c("uniform", "normal"))
  d_sqrt_inv = 1 / sqrt(d)
  if (initialization == "uniform") {
    return(nn_init_uniform_(x, a = -d_sqrt_inv, b = d_sqrt_inv))
  } else {
    return(nn_init_normal_(x, std=d_sqrt_inv))
  }
}

nn_cls_token = nn_module(
  "nn_cls_token",
  initialize = function(d_token, initialization) {
    self$d_token = checkmate::assert_integerish(d_token,
                                                lower = 1L, any.missing = FALSE, len = 1,
                                                coerce = TRUE
    )
    self$weight = nn_parameter(torch_empty(d_token))
    self$initialization = initialization
    self$reset_parameters()
  },
  reset_parameters = function() {
    initialize_token_(self$weight, d = self$d_token, self$initialization)
  },
  expand = function(...) {
    leading_dimensions = list(...)
    if(length(leading_dimensions) == 0) {
      return(self$weight)
    }
    new_dims = rep(1, length(leading_dimensions) - 1)
    return(self$weight$view(c(new_dims, -1))$expand(c(leading_dimensions, -1)))
  },
  forward = function(input) {
    return(torch_cat(list(input, self$expand(input$shape[1], 1)), dim=2)) # the length of tensor, multiplies all dimensions
  }
)