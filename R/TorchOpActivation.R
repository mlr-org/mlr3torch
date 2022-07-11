#' @title Torch Activation Function
#' @description
#' Common activation functions. They can also be accessed directly via their `"id"` through
#' the short-hand constructor `"top"`.
#'
#' @section Parameters:
#' If the value of the `activation` constructor argument is set to one of the available activation
#' functions (see `torch_reflections$activation`). The parameter set is dynamically constructed and
#' set to the parameters of the activation functions.
#'
#' If left as `NULL`, the parameters are set to `fn` and `args`:
#' * `fn` `character(1)`\cr
#'   The choice of the activation function, see `torch_reflections$activation`.
#' * `args`:: `list`\cr
#'   A list with arguments for the actication function.
#'
#' @template param_id
#' @template param_param_vals
#' @param activation (`character(1)`)\cr The activation function, see `torch_reflections$activation`.
#'
#' @examples
#' top("relu")
#' # is the same as
#' top("activation", activation = "relu")
#'
#' @export
TorchOpActivation = R6Class("TorchOpActivation",
  inherit = TorchOp,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(id = ifelse(is.null(activation), "activation", activation),
      param_vals = list(), activation = NULL) {
      assert_choice(activation, torch_reflections$activation, null.ok = TRUE)
      private$.activation = activation

      if (!is.null(activation)) {
        param_set = paramsets_activation$get(activation)
      } else {
        param_set = ps(
          fn = p_fct(levels = torch_reflections$activation, tags = c("train", "required")),
          args = p_uty(tags = "train", custom_check = check_activation_args)
        )
      }
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals
      )
    }
  ),
  private = list(
    .build = function(inputs, task) {
      if (!is.null(private$.activation)) {
        pv = self$param_set$get_values(tag = "train")
        layer = invoke(get_activation(private$.activation), .args = pv)
        return(layer)
      }
      pv = self$param_set$get_values(tag = "train")

      invoke(get_activation(pv$fn), .args = pv$args)
    },
    .activation = NULL
  )
)

#' @include mlr_torchops.R
mlr_torchops$add("activation", TorchOpActivation)

make_torchop_activation = function(activation) {
  mlr_torchops$add(
    activation,
    function(id = activation, param_vals = list()) {
      TorchOpActivation$new(id = id, param_vals = param_vals, activation = activation)
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
