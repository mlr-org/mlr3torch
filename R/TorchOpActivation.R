#' @title Torch Activation Function
#' @description
#' Class for Torch activation functions
#' Don't use this class directly but the corresponding object.
#' @export
TorchOpActivationAbstract = R6Class("TorchOpActivationAbstract",
  inherit = TorchOp,
  public = list(
    #' @description Initializes an instance of this [R6][R6::R6Class] class.
    #' @param id (`character(1)`)\cr
    #'   The id for of the object.
    #' @param param_vals (named `list()`)\cr
    #'   The initial parameters for the object.
    #' @param .act (`character(1)`)\cr
    #'   The activation function (see `torch_reflections$activation`).
    initialize = function(id = .act, param_vals = list(), .act) {
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
    .build = function(inputs, param_vals, task) {
      constructor = get_activation(private$.act)
      invoke(constructor, .args = param_vals)
    },
    .act = NULL
  )
)

#' @title Activation Functions
#' @description
#' This is implements all generic activation function.
#' @export
TorchOpActivation = R6Class("TorchOpActivation",
  inherit = TorchOp,
  public = list(
    initialize = function(id = "activation", param_vals = list()) {
      param_set = ps(
        activation = p_fct(levels = torch_reflections$activation, tags = c("train", "required")),
        activation_args = p_uty(tags = "train")
      )

      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals
      )
    }
  ),
  private = list(
    .build = function(inputs, param_vals, task) {
      aa = self$param_set$values$activation_args
      a = self$param_set$values$activation
      aclass = get_activation(a)

      invoke(aclass, .args = aa)
    }
  )
)

#'
mlr_torchops$add("activation", TorchOpActivation)

make_torchop_activation = function(act) {
  mlr_torchops$add(
    act,
    function(id = act, param_vals = list()) {
      TorchOpActivationAbstract$new(id = id, param_vals = param_vals, .act = act)
    }
  )
}

make_torchop_activation("elu")
make_torchop_activation("hardshrink")
make_torchop_activation("hardsigmoid")
make_torchop_activation("hardtanh")
make_torchop_activation("hardswish")
make_torchop_activation("leaky_relu")
make_torchop_activation("log_sigmoid")
make_torchop_activation("prelu")
make_torchop_activation("relu")
make_torchop_activation("relu6")
make_torchop_activation("rrelu")
make_torchop_activation("selu")
make_torchop_activation("sigmoid")
make_torchop_activation("softplus")
make_torchop_activation("softshrink")
make_torchop_activation("softsign")
make_torchop_activation("tanh")
make_torchop_activation("tanhshrink")
make_torchop_activation("threshold")
make_torchop_activation("glu")
