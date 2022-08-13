
PipeOpTorchActivation = R6Class("PipeOpTorchActivation",
  inherit = PipeOpTorch
)

#' @include zzz.R
make_activation = function(name, param_set, parent_env = parent.frame()) {
  classname = paste0("PipeOpTorchActivation", capitalize(name))
  idname = paste0("activation_", name)

  result <- R6Class(classname,
    inherit = PipeOpTorchActivation,
    public = list(
      initialize = eval(substitute(function(id = idname, param_vals = list()) {
        param_set = ps
        super$initialize(
          id = id,
          param_set = param_set,
          param_vals = param_vals,
          module_generator = module
        )
      },
      list(idname = idname, ps = substitute(param_set), module = as.symbol(paste0("nn_", name)))),
      envir = parent_env)
    ),
    parent_env = parent_env
  )
#  eval(substitute(register_po(idname, constructor = result), list(idname = idname, result = as.symbol(classname))))
  register_po(idname, result)
  result
}

#' @export
PipeOpTorchActivationElu = make_activation("elu", param_set = ps(
  alpha = p_dbl(default = 1, tags = "train"),
  inplace = p_lgl(default = FALSE, tags = "train")
))

#' @export
PipeOpTorchActivationHardShrink = make_activation("hardshrink", param_set = ps(
  lambd = p_dbl(default = 0.5, tags = "train")
))

#' @export
PipeOpTorchActivationHardSigmoid = make_activation("hardsigmoid", param_set = ps())

#' @export
PipeOpTorchActivationHardTanh = make_activation("hardtanh", param_set = ps(
  min_val = p_dbl(default = -1, tags = "train"),
  max_val = p_dbl(default = 1, tags = "train"),
  inplace = p_lgl(default = FALSE, tags = "train")
))

#' @export
PipeOpTorchActivationHardSwish = make_activation("hardswish", param_set = ps())

#' @export
PipeOpTorchActivationLeakyRelu = make_activation("leaky_relu", param_set = ps(
  negative_slope = p_dbl(default = 0.01, tags = "train"),
  inplace = p_lgl(default = FALSE, tags = "train")
))

#' @export
PipeOpTorchActivationLogSigmoid = make_activation("log_sigmoid", param_set = ps())

#' @export
PipeOpTorchActivationPrelu = make_activation("prelu", param_set = ps(
  num_parameters = p_int(1, tags = "train"),
  init = p_dbl(default = 0.25, tags = "train")
))

#' @export
PipeOpTorchActivationRelu = make_activation("relu", param_set = ps(
  inplace = p_lgl(default = FALSE, tags = "train")
))

#' @export
PipeOpTorchActivationRelu6 = make_activation("relu6", param_set = ps(
  inplace = p_lgl(default = FALSE, tags = "train")
))

#' @export
PipeOpTorchActivationRrelu = make_activation("rrelu", param_set = ps(
  lower = p_dbl(default = 1 / 8, tags = "train"),
  upper = p_dbl(default = 1 / 3, tags = "train"),
  inplace = p_lgl(default = FALSE, tags = "train")
))

#' @export
PipeOpTorchActivationSelu = make_activation("selu", param_set = ps(
  inplace = p_lgl(tags = "train")
))

#' @export
PipeOpTorchActivationSelu = make_activation("celu", param_set = ps(
  alpha = p_dbl(default = 1.0, tags = "train"),
  inplace = p_lgl(default = FALSE, tags = "train")
))

#' @export
PipeOpTorchActivationSigmoid = make_activation("gelu", param_set = ps())

#' @export
PipeOpTorchActivationSigmoid = make_activation("sigmoid", param_set = ps())

#' @export
PipeOpTorchActivationSoftPlus = make_activation("softplus", param_set = ps(
  beta = p_dbl(default = 1, tags = "train"),
  threshold = p_dbl(default = 20, tags = "train")
))

#' @export
PipeOpTorchActivationSoftShrink = make_activation("softshrink", param_set = ps(
  lambd = p_dbl(default = 0.5, upper = 1, tags = "train")
))

#' @export
PipeOpTorchActivationSoftSign = make_activation("softsign", param_set = ps())

#' @export
PipeOpTorchActivationTanh = make_activation("tanh", param_set = ps())

#' @export
PipeOpTorchActivationTanhShrink = make_activation("tanhshrink", param_set = ps())

#' @export
PipeOpTorchActivationThreshold = make_activation("threshold", param_set = ps(
  threshold = p_dbl(tags = "train"),
  value = p_dbl(tags = "train"),
  inplace = p_lgl(default = FALSE, tags = "train")
))

#' @export
PipeOpTorchActivationGlu = make_activation("glu", param_set = ps(
  dim = p_int(default = -1L, lower = 1L, tags = "train", special_vals = list(-1L))
))
