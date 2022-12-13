#' @export
PipeOpTorchOptimizer = R6Class("PipeOpTorchOptimizer",
  inherit = PipeOpTorch,
  public = list(
    initialize = function(optimizer, id = "torch_optimizer", param_vals = list()) {
      private$.optimizer = assert_r6(as_torch_optimizer(optimizer), "TorchOptimizer")
      super$initialize(
        id = id,
        param_set = alist(private$.optimizer$param_set),
        param_vals = param_vals,
        module_generator = NULL,
        inname = "input"
      )
    }
  ),
  private = list(
    .train = function(inputs) {
      expect_true(is.null(inputs[[1L]]$optimizer))
      inputs[[1]]$optimizer = private$.optimizer$clone(deep = TRUE)
      inputs
    },
    .optimizer = NULL
  )
)

#' @include zzz.R
register_po("torch_optimizer", PipeOpTorchOptimizer)
