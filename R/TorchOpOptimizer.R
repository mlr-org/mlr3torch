#' @export
TorchOpOptimizer = R6Class("TorchOpOptimizer",
  inherit = PipeOpTorch,
  public = list(
    initialize = function(optimizer, id = "optimizer", param_vals = list()) {
      assert_r6(optimizer, "TorchOptimizer")
      private$.optimizer = optimizer
      super$initialize(
        id = id,
        param_set = alist(private$.optimizer$param_set),
        param_vals = param_vals
      )
    }
  ),
  private = list(
    .train = function(inputs) {
      inputs[[1]]$optimizer = private$.optimizer$clone(deep = TRUE)
      inputs
    },
    .shapes_out = function(shapes_in, param_vals) shapes_in,
    .shape_dependent_params = function(shapes_in) list(),
    .optimizer = NULL
  )
)

#' @include mlr_torchops.R
mlr_torchops$add("optimizer", TorchOpOptimizer)
