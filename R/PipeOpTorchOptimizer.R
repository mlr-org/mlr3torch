#' @export
PipeOpTorchOptimizer = R6Class("PipeOpTorchOptimizer",
  inherit = PipeOpTorch,
  public = list(
    initialize = function(optimizer, id = "torch_optimizer", param_vals = list()) {
      assert_r6(optimizer, "PipeOpTorchtimizer")
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
    .optimizer = NULL
  )
)

#' @include zzz.R
register_po("torch_optimizer", PipeOpTorchOptimizer)
