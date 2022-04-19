#' @export
TorchOpActivation = R6Class("TorchOpActivation",
  inherit = TorchOp,
  public = list(
    initialize = function(id = .activation, param_vals = list(), .activation) {
      assert_choice(.activation, torch_reflections$activation)
      private$.activation = .activation
      param_set = make_paramset_activation(.activation)
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals
      )
    }
  ),
  private = list(
    .operator = "activation",
    .build = function(inputs, param_vals, task, y) {
      constructor = get_activation(private$.activation)
      invoke(constructor, .args = param_vals)
    },
    .activation = NULL
  )
)

mlr_torchops$add("activation", TorchOpActivation)
