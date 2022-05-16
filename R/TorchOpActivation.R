#' @title Torch Activation Function
#' @export
TorchOpActivation = R6Class("TorchOpActivation",
  inherit = TorchOp,
  public = list(
    #' @description Initializes an instance of this [R6][R6::R6Class] class.
    #' @param id (`character(1)`)\cr
    #'   The id for of the object.
    #' @parm param_vals (named `list()`)\cr
    #'   The initial parameters for the object.
    initialize = function(id = .activation, param_vals = list(), .act) {
      assert_choice(.act, torch_reflections$activation)
      private$.act = .act
      param_set = paramsets_activation$get(.act)
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
      constructor = get_activation(private$.act)
      invoke(constructor, .args = param_vals)
    },
    .act = NULL
  )
)

make_torchop_activation = function(classname, act) {
  class = R6Class(sprintf("TorchOp%s", classname),
    inherit = TorchOpActivation,
    public = list(
      initialize = function(id = act, param_vals = list(), .act = act) {
        param_set = paramsets_activation$get(.act)
        super$initialize(
          id = id,
          param_vals = param_vals,
          .act = act
        )
      }
    )
  )
  #' @include mlr_torchops.R
  mlr_torchops$add(act, class)
  return(class)
}

#' @export
TorchOpElu = make_torchop_activation("Elu", "elu")
#' @export
TorchOpElu = make_torchop_activation("HardShrink", "hardshrink")
#' @export
TorchOpElu = make_torchop_activation("HardSigmoid", "hardsigmoid")
#' @export
TorchOpElu = make_torchop_activation("HardTanh", "hardtanh")
#' @export
TorchOpElu = make_torchop_activation("HardSwish", "hardwish")
#' @export
TorchOpElu = make_torchop_activation("LeakyRelu", "leaky_relu")
#' @export
TorchOpElu = make_torchop_activation("LogSigmoid", "log_sigmoid")
#' @export
TorchOpElu = make_torchop_activation("PReLU", "prelu")
#' @export
TorchOpElu = make_torchop_activation("ReLU", "relu")
#' @export
TorchOpElu = make_torchop_activation("ReLU6", "relu6")
#' @export
TorchOpElu = make_torchop_activation("RReLU", "rrelu")
#' @export
TorchOpElu = make_torchop_activation("SeLU", "selu")
#' @export
TorchOpElu = make_torchop_activation("Sigmoid", "sigmoid")
#' @export
TorchOpElu = make_torchop_activation("SoftPlus", "softplus")
#' @export
TorchOpElu = make_torchop_activation("SoftShrink", "softshrink")
#' @export
TorchOpElu = make_torchop_activation("SoftSign", "softsign")
#' @export
TorchOpElu = make_torchop_activation("Tanh", "tanh")
#' @export
TorchOpElu = make_torchop_activation("TanhShrink", "tanhshrink")
#' @export
TorchOpElu = make_torchop_activation("Threshold", "threshold")
#' @export
TorchOpElu = make_torchop_activation("GLU", "glu")

