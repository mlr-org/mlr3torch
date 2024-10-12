PipeOpTorchAdaptiveAvgPool = R6Class("PipeOpTorchAdaptiveAvgPool",
  inherit = PipeOpTorch,
  public = list(
    # @description Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(id, d, output_size) {
      # WIP: temporarily only allow 1d
      private$.d = assert_int(d, lower = 1, upper = 1)
      module_generator = switch(d, nn_adaptive_avg_pool1d)
      check_vector = make_check_vector(private$.d)
      param_set = 
      
      super$initialize(
        id = id,
        module_generator = module_generator
      )
    }
  ),
  private = list(
    .shapes_
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
    initialize = function(id = "nn_adaptive_avg_pool1d", output_size) {
      super$initialize(id = id, d = 1, output_size = output_size)
    }
  )
)
