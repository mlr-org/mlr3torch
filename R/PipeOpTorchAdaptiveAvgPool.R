PipeOpTorchAdaptiveAvgPool = R6Class("PipeOpTorchAdaptiveAvgPool",
  inherit = PipeOpTorch,
  public = list(
    # @description Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(id, d, param_vals = list()) {
      # TODO: allow higher dimensions
      private$.d = assert_int(d, lower = 1, upper = 1)
      module_generator = switch(d, nn_adaptive_avg_pool1d)
      check_vector = make_check_vector(private$.d)
      param_set = ps(
        output_size = p_uty(custom_check = check_vector, tags = c("required", "train"))
      )
      
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals,
        module_generator = module_generator
      )
    }
  ),
  private = list(
    .shapes_out = function(shapes_in, param_vals, task) {
      list(output_size = param_vals$output_size)
    },
    .d = NULL
  )
)

#' @title 1D Adaptive Average Pooling
#' 
#' @templateVar id nn_adaptive_avg_pool1d
#' 
#' @inherit torch::nnf_adaptive_avg_pool1d description
#' 
#' @section Parameters:
#' * `output_size` :: `integer()`\cr
#'   The target output size. A single number.
#' 
#' @section Internals:
#' Calls [`nn_adaptive_avg_pool1d()`][torch::nn_adaptive_avg_pool1d] during training.
#' @export
PipeOpTorchAdaptiveAvgPool1D = R6Class("PipeOpTorchAdaptiveAvgPool1D", inherit = PipeOpTorchAdaptiveAvgPool,
  public = list(
    #' @description Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    initialize = function(id = "nn_adaptive_avg_pool1d", param_vals = list()) {
      super$initialize(id = id, d = 1, param_vals = param_vals)
    }
  )
)

#' @include zzz.R
register_po("nn_adaptive_avg_pool1d", PipeOpTorchAdaptiveAvgPool1D)