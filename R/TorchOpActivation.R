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
TorchOpHardShrink = make_torchop_activation("HardShrink", "hardshrink")
#' @export
TorchOpHardSigmoid = make_torchop_activation("HardSigmoid", "hardsigmoid")
#' @export
TorchOpHardTanh = make_torchop_activation("HardTanh", "hardtanh")
#' @export
TorchOpHardSwish = make_torchop_activation("HardSwish", "hardswish")
#' @export
TorchOpLeakyReLU = make_torchop_activation("LeakyReLU", "leaky_relu")
#' @export
TorchOpLogSigmoid = make_torchop_activation("LogSigmoid", "log_sigmoid")
#' @export
TorchOpPReLU = make_torchop_activation("PReLU", "prelu")
#' @export
TorchOpReLU = make_torchop_activation("ReLU", "relu")
#' @export
TorchOpReLU6 = make_torchop_activation("ReLU6", "relu6")
#' @export
TorchOpRReLU = make_torchop_activation("RReLU", "rrelu")
#' @export
TorchOpSeLU = make_torchop_activation("SeLU", "selu")
#' @export
TorchOpSigmoid = make_torchop_activation("Sigmoid", "sigmoid")
#' @export
TorchOpSoftPlus = make_torchop_activation("SoftPlus", "softplus")
#' @export
TorchOpSoftShrink = make_torchop_activation("SoftShrink", "softshrink")
#' @export
TorchOpSoftSign = make_torchop_activation("SoftSign", "softsign")
#' @export
TorchOpTanh = make_torchop_activation("Tanh", "tanh")
#' @export
TorchOpTanhShrink = make_torchop_activation("TanhShrink", "tanhshrink")
#' @export
TorchOpThreshold = make_torchop_activation("Threshold", "threshold")
#' @export
TorchOpGLU = make_torchop_activation("GLU", "glu")

