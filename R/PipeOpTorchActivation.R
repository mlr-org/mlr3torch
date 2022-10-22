#' @include zzz.R
make_activation = function(name, class, param_set, parent_env = parent.frame()) {
  classname = paste0("PipeOpTorch", class)
  idname = paste0("nn_", name)

  ps = substitute(param_set)
  module = as.symbol(paste0("nn_", name))

  init_fun_proto = function(id = idname, param_vals = list()) {
    param_set = ps
    super$initialize(
      id = id,
      param_set = param_set,
      param_vals = param_vals,
      module_generator = module,
      tags = "activation"
    )
  }

  # uber-hacky, but document() won't work otherwise
  init_fun = init_fun_proto
  formals(init_fun) = pairlist(id = idname, param_vals = list())
  body(init_fun)[[2]][[3]] = ps
  body(init_fun)[[3]][[5]] = module
  attributes(init_fun) = attributes(init_fun_proto)

  result <- R6Class(classname,
    inherit = PipeOpTorch,
    public = list(initialize = init_fun),
    parent_env = parent_env
  )
#  eval(substitute(register_po(idname, constructor = result), list(idname = idname, result = as.symbol(classname))))
  register_po(idname, result)
  result
}

#' @title ELU Activation Function
#'
#' @usage NULL
#' @name mlr_pipeops_torch_elu
#' @format `r roxy_pipeop_torch_format()`
#'
#' @inherit torch::nnf_elu description
#'
#' @section Construction: `r roxy_pipeop_torch_construction("ELU")`
#' `r roxy_param_id("nn_elu")`
#' `r roxy_param_param_vals()`
#'
#' @section Input and Output Channels: `r roxy_pipeop_torch_channels_default()`
#' @section State: `r roxy_pipeop_torch_state_default()`
#'
#' @section Parameters:
#' * `alpha` :: `numeric(1)`\cr
#'   The alpha value for the ELU formulation. Default: 1.0
#' * `inplace` :: `logical(1)`\cr
#'   Whether to do the operation in-place. Default: `FALSE`.
#'
#' @section Internals: Calls [`torch::nn_elu()`] when trained.
#' @section Fields: `r roxy_pipeop_torch_fields_default()`
#' @section Methods: `r roxy_pipeop_torch_methods_default()`
#' @section Credit: `r roxy_pipeop_torch_license()`
#' @family PipeOpTorch
#' @export
PipeOpTorchELU = make_activation("elu", "ELU", param_set = ps(
  alpha = p_dbl(default = 1, tags = "train"),
  inplace = p_lgl(default = FALSE, tags = "train")
))

#' @title Hard Shrink Activation Function
#'
#' @usage NULL
#' @name mlr_pipeops_torch_hard_shrink
#' @format `r roxy_pipeop_torch_format()`
#'
#' @inherit torch::nnf_hardshrink description
#'
#' @section Construction: `r roxy_pipeop_torch_construction("HardShrink")`
#' `r roxy_param_id("nn_hardshrink")`
#' `r roxy_param_param_vals()`
#'
#' @section Input and Output Channels: `r roxy_pipeop_torch_channels_default()`
#' @section State: `r roxy_pipeop_torch_state_default()`
#'
#' @section Parameters:
#' * `lambd` :: `numeric(1)`\cr
#'   The $\lambda$ value for the Hardshrink formulation formulation. Default 0.5.
#'
#' @section Internals: Calls [`torch::nn_hardshrink()`] when trained.
#' @section Fields: `r roxy_pipeop_torch_fields_default()`
#' @section Methods: `r roxy_pipeop_torch_methods_default()`
#' @section Credit: `r roxy_pipeop_torch_license()`
#' @family PipeOpTorch
#' @export
PipeOpTorchHardShrink = make_activation("hardshrink", "HardShrink", param_set = ps(
  lambd = p_dbl(default = 0.5, tags = "train")
))

#' @title Hard Sigmoid Activation Function
#'
#' @usage NULL
#' @name mlr_pipeops_torch_hardsigmoid
#' @format `r roxy_pipeop_torch_format()`
#'
#' @inherit torch::nnf_hardsigmoid description
#'
#' @section Construction: `r roxy_pipeop_torch_construction("HardSigmoid")`
#' `r roxy_param_id("nn_hardsigmoid")`
#' `r roxy_param_param_vals()`
#'
#' @section Input and Output Channels: `r roxy_pipeop_torch_channels_default()`
#' @section State: `r roxy_pipeop_torch_state_default()`
#'
#' @section Parameters:
#' No parameters.
#'
#' @section Internals: Calls [`torch::nn_hardsigmoid()`] when trained.
#' @section Fields: `r roxy_pipeop_torch_fields_default()`
#' @section Methods: `r roxy_pipeop_torch_methods_default()`
#' @section Credit: `r roxy_pipeop_torch_license()`
#' @family PipeOpTorch
#' @export
PipeOpTorchHardSigmoid = make_activation("hardsigmoid", "HardSigmoid", param_set = ps())

#' @title Hard Tanh Activation Function
#'
#' @usage NULL
#' @name mlr_pipeops_torch_hard
#' @format `r roxy_pipeop_torch_format()`
#'
#' @inherit torch::nnf_hardtanh description
#'
#' @section Construction: `r roxy_pipeop_torch_construction("HardTanh")`
#' `r roxy_param_id("nn_hardtanh")`
#' `r roxy_param_param_vals()`
#'
#' @section Input and Output Channels: `r roxy_pipeop_torch_channels_default()`
#' @section State: `r roxy_pipeop_torch_state_default()`
#'
#' @section Parameters:
#' * `min_val` :: `numeric(1)`\cr
#'   Minimum value of the linear region range. Default: -1.
#' * `max_val` :: `numeric(1)`\cr
#'   Maximum value of the linear region range. Default: 1.
#' * `inplace` :: `logical(1)`\cr
#'   Can optionally do the operation in-place. Default: `FALSE`.
#'
#' @section Internals:
#' Calls [`torch::nn_hardtanh()`] when trained.
#'
#' @section Fields: `r roxy_pipeop_torch_fields_default()`
#' @section Methods: `r roxy_pipeop_torch_methods_default()`
#' @section Credit:
#' `r roxy_pipeop_torch_license()`
#'
#' @family PipeOpTorch
#' @export
PipeOpTorchHardTanh = make_activation("hardtanh", "HardTanh", param_set = ps(
  min_val = p_dbl(default = -1, tags = "train"),
  max_val = p_dbl(default = 1, tags = "train"),
  inplace = p_lgl(default = FALSE, tags = "train")
))

# Not Yet Implemented
# #' @title Hard Swish Activation Function
# #'
# #' @usage NULL
# #' @name mlr_pipeops_torch_hardswishardswish
# #' @format `r roxy_pipeop_torch_format()`
# #'
# #' @inherit torch::nnf_hardwish description
# #'
# #' @section Construction: `r roxy_pipeop_torch_construction("HardSwish")`
# #' `r roxy_param_id("nn_hardswish")`
# #' `r roxy_param_param_vals()`
# #'
# #' @section Input and Output Channels: `r roxy_pipeop_torch_channels_default()`
# #'
# #' @section State: `r roxy_pipeop_torch_state_default()`
# #'
# #' @section Parameters:
# #' No parameters.
# #'
# #' @section Internals: Calls [`torch::nn_hardswish()`] when trained.
# #' @section Fields: `r roxy_pipeop_torch_fields_default()`
# #' @section Methods: `r roxy_pipeop_torch_methods_default()`
# #' @section Credit: `r roxy_pipeop_torch_license()`
# #' @family PipeOpTorch
# #' @export
# PipeOpTorchHardSwish = make_activation("hardswish", "HardSwish", param_set = ps())

#' @title Leaky ReLU Activation Function
#'
#' @usage NULL
#' @name mlr_pipeops_torch_leaky_relu
#' @format `r roxy_pipeop_torch_format()`
#'
#' @inherit torch::nnf_leaky_relu description
#'
#' @section Construction: `r roxy_pipeop_torch_construction("LeakyReLU")`
#' `r roxy_param_id("nn_leaky_relu")`
#' `r roxy_param_param_vals()`
#'
#' @section Input and Output Channels: `r roxy_pipeop_torch_channels_default()`
#' @section State: `r roxy_pipeop_torch_state_default()`
#'
#' @section Parameters:
#' * `negative_slope` :: `numeric(1)`\cr
#'   Controls the angle of the negative slope. Default: 1e-2.
#' * `inplace` :: `logical(1)`\cr
#'   Can optionally do the operation in-place. Default: ‘FALSE’.
#'
#' @section Internals: Calls [`torch::nn_hardswish()`] when trained.
#' @section Fields: `r roxy_pipeop_torch_fields_default()`
#' @section Methods: `r roxy_pipeop_torch_methods_default()`
#' @section Credit: `r roxy_pipeop_torch_license()`
#' @family PipeOpTorch
#' @export
PipeOpTorchLeakyReLU = make_activation("leaky_relu", "LeakyReLU", param_set = ps(
  negative_slope = p_dbl(default = 0.01, tags = "train"),
  inplace = p_lgl(default = FALSE, tags = "train")
))

#' @title Log Sigmoid Activation Function
#'
#' @usage NULL
#' @name mlr_pipeops_torch_log_sigmoid
#' @format `r roxy_pipeop_torch_format()`
#'
#' @inherit torch::nnf_logsigmoid description
#'
#' @section Construction: `r roxy_pipeop_torch_construction("LogSigmoid")`
#' `r roxy_param_id("nn_log_sigmoid")`
#' `r roxy_param_param_vals()`
#'
#' @section Input and Output Channels: `r roxy_pipeop_torch_channels_default()`
#' @section State: `r roxy_pipeop_torch_state_default()`
#'
#' @section Parameters:
#' No parameters.
#'
#' @section Fields: `r roxy_pipeop_torch_fields_default()`
#' @section Methods: `r roxy_pipeop_torch_methods_default()`
#' @section Internals: Calls [`torch::nn_log_sigmoid()`] when trained.
#' @section Credit: `r roxy_pipeop_torch_license()`
#' @family PipeOpTorch
#' @export
PipeOpTorchLogSigmoid = make_activation("log_sigmoid", "LogSigmoid", param_set = ps())

#' @title PReLU Activation Function
#'
#' @usage NULL
#' @name mlr_pipeops_torch_prelu
#' @format `r roxy_pipeop_torch_format()`
#'
#' @inherit torch::nnf_prelu description
#'
#' @section Construction: `r roxy_pipeop_torch_construction("PReLU")`
#' `r roxy_param_id("nn_prelu")`
#' `r roxy_param_param_vals()`
#'
#' @section Input and Output Channels: `r roxy_pipeop_torch_channels_default()`
#' @section State: `r roxy_pipeop_torch_state_default()`
#'
#' @section Parameters:
#' * `num_parameters` :: `integer(1)`:
#'   Number of a to learn. Although it takes an int as input, there is only two values are legitimate: 1, or the
#'   number of channels at input. Default: 1.
#' * `init` :: `numeric(1)`\cr T
#'   The initial value of a. Default: 0.25.
#'
#' @section Fields: `r roxy_pipeop_torch_fields_default()`
#' @section Methods: `r roxy_pipeop_torch_methods_default()`
#' @section Internals: Calls [`torch::nn_prelu()`] when trained.
#' @section Credit: `r roxy_pipeop_torch_license()`
#' @family PipeOpTorch
#' @export
PipeOpTorchPReLU = make_activation("prelu", "PReLU", param_set = ps(
  num_parameters = p_int(1, tags = "train"),
  init = p_dbl(default = 0.25, tags = "train")
))

#' @title ReLU Activation Function
#' @usage NULL
#' @name mlr_pipeops_torch_relu
#' @format `r roxy_pipeop_torch_format()`
#'
#' @inherit torch::nnf_relu description
#'
#' @section Construction: `r roxy_pipeop_torch_construction("ReLU")`
#' `r roxy_param_id("nn_relu")`
#' `r roxy_param_param_vals()`
#'
#' @section Input and Output Channels: `r roxy_pipeop_torch_channels_default()`
#' @section State: `r roxy_pipeop_torch_state_default()`
#'
#' @section Parameters:
#' * `inplace` :: `logical(1)`\cr
#'   Whether to do the operation in-place. Default: `FALSE`.
#'
#' @section Fields: `r roxy_pipeop_torch_fields_default()`
#' @section Methods: `r roxy_pipeop_torch_methods_default()`
#' @section Internals: Calls [`torch::nn_relu()`] when trained.
#' @section Credit: `r roxy_pipeop_torch_license()`
#' @family PipeOpTorch
#' @export
PipeOpTorchReLU = make_activation("relu", "ReLU", param_set = ps(
  inplace = p_lgl(default = FALSE, tags = "train")
))

#' @title Relu6 Activation Function
#'
#' @usage NULL
#' @name mlr_pipeops_torch_relu6
#' @format `r roxy_pipeop_torch_format()`
#'
#' @inherit torch::nnf_relu6 description
#'
#' @section Construction: `r roxy_pipeop_torch_construction("Relu6")`
#' `r roxy_param_id("nn_relu6")`
#' `r roxy_param_param_vals()`
#'
#' @section Input and Output Channels: `r roxy_pipeop_torch_channels_default()`
#' @section State: `r roxy_pipeop_torch_state_default()`
#'
#' @section Parameters:
#' * `inplace` :: `logical(1)`\cr
#'   Whether to do the operation in-place. Default: `FALSE`.
#'
#' @section Fields: `r roxy_pipeop_torch_fields_default()`
#' @section Methods: `r roxy_pipeop_torch_methods_default()`
#' @section Internals: Calls [`torch::nn_relu6()`] when trained.
#' @section Credit: `r roxy_pipeop_torch_license()`
#' @family PipeOpTorch
#' @export
PipeOpTorchRelu6 = make_activation("relu6", "ReLU6", param_set = ps(
  inplace = p_lgl(default = FALSE, tags = "train")
))

#' @title RReLU Activation Function
#'
#' @usage NULL
#' @name mlr_pipeops_torch_rrelu
#' @format `r roxy_pipeop_torch_format()`
#'
#' @inherit torch::nnf_rrelu description
#'
#' @section Construction: `r roxy_pipeop_torch_construction("RReLU")`
#' `r roxy_param_id("nn_rrelu")`
#' `r roxy_param_param_vals()`
#'
#' @section Input and Output Channels: `r roxy_pipeop_torch_channels_default()`
#' @section State: `r roxy_pipeop_torch_state_default()`
#'
#' @section Parameters:
#' * `lower`:: `numeric(1)`\cr
#'   Lower bound of the uniform distribution. Default: 1/8.
#' * `upper`:: `numeric(1)`\cr
#'   Upper bound of the uniform distribution. Default: 1/3.
#' * `inplace` :: `logical(1)`\cr
#'   Whether to do the operation in-place. Default: `FALSE`.
#'
#' @section Fields: `r roxy_pipeop_torch_fields_default()`
#' @section Methods: `r roxy_pipeop_torch_methods_default()`
#' @section Internals: Calls [`torch::nn_rrelu()`] when trained.
#' @section Credit: `r roxy_pipeop_torch_license()`
#' @family PipeOpTorch
#' @export
PipeOpTorchRReLU = make_activation("rrelu", "RReLU", param_set = ps(
  lower = p_dbl(default = 1 / 8, tags = "train"),
  upper = p_dbl(default = 1 / 3, tags = "train"),
  inplace = p_lgl(default = FALSE, tags = "train")
))

#' @title SELU Activation Function
#'
#' @usage NULL
#' @name mlr_pipeops_torch_selu
#' @format `r roxy_pipeop_torch_format()`
#'
#' @inherit torch::nnf_selu description
#'
#' @section Construction: `r roxy_pipeop_torch_construction("SELU")`
#' `r roxy_param_id("nn_selu")`
#' `r roxy_param_param_vals()`
#'
#' @section Input and Output Channels: `r roxy_pipeop_torch_channels_default()`
#' @section State: `r roxy_pipeop_torch_state_default()`
#'
#' @section Parameters:
#' * `inplace` :: `logical(1)`\cr
#'   Whether to do the operation in-place. Default: `FALSE`.
#'
#' @section Fields: `r roxy_pipeop_torch_fields_default()`
#' @section Methods: `r roxy_pipeop_torch_methods_default()`
#' @section Internals: Calls [`torch::nn_selu()`] when trained.
#' @section Credit: `r roxy_pipeop_torch_license()`
#' @family PipeOpTorch
#' @export
PipeOpTorchSELU = make_activation("selu", "SELU", param_set = ps(
  inplace = p_lgl(tags = "train")
))

#' @title CELU Activation Function
#'
#' @usage NULL
#' @name mlr_pipeops_torch_celu
#' @format `r roxy_pipeop_torch_format()`
#'
#' @inherit torch::nnf_celu description
#'
#' @section Construction: `r roxy_pipeop_torch_construction("CELU")`
#' `r roxy_param_id("nn_celu")`
#' `r roxy_param_param_vals()`
#'
#' @section Input and Output Channels: `r roxy_pipeop_torch_channels_default()`
#' @section State: `r roxy_pipeop_torch_state_default()`
#'
#' @section Parameters:
#' * `alpha` :: `numeric(1)`\cr
#'   The alpha value for the ELU formulation. Default: 1.0
#' * `inplace` :: `logical(1)`\cr
#'   Whether to do the operation in-place. Default: `FALSE`.
#'
#' @section Fields: `r roxy_pipeop_torch_fields_default()`
#' @section Methods: `r roxy_pipeop_torch_methods_default()`
#' @section Internals: Calls [`torch::nn_celu()`] when trained.
#' @section Credit: `r roxy_pipeop_torch_license()`
#' @family PipeOpTorch
#' @export
PipeOpTorchCelu = make_activation("celu", "CELU", param_set = ps(
  alpha = p_dbl(default = 1.0, tags = "train"),
  inplace = p_lgl(default = FALSE, tags = "train")
))

#' @title GELU Activation Function
#'
#' @usage NULL
#' @name mlr_pipeops_torch_gelu
#' @format `r roxy_pipeop_torch_format()`
#'
#' @inherit torch::nnf_gelu description
#'
#' @section Construction: `r roxy_pipeop_torch_construction("GELU")`
#' `r roxy_param_id("nn_gelu")`
#' `r roxy_param_param_vals()`
#'
#' @section Input and Output Channels: `r roxy_pipeop_torch_channels_default()`
#' @section State: `r roxy_pipeop_torch_state_default()`
#'
#' @section Parameters:
#' No parameters.
#'
#' @section Fields: `r roxy_pipeop_torch_fields_default()`
#' @section Methods: `r roxy_pipeop_torch_methods_default()`
#' @section Internals: Calls [`torch::nn_gelu()`] when trained.
#' @section Credit: `r roxy_pipeop_torch_license()`
#' @family PipeOpTorch
#' @export
PipeOpTorchGELU = make_activation("gelu", "GELU", param_set = ps())

#' @title Sigmoid Activation Function
#'
#' @usage NULL
#' @name mlr_pipeops_torch_sigmoid
#' @format `r roxy_pipeop_torch_format()`
#'
#' @inherit torch::nnf_sigmoid description
#'
#' @section Construction: `r roxy_pipeop_torch_construction("Sigmoid")`
#' `r roxy_param_id("nn_sigmodi")`
#' `r roxy_param_param_vals()`
#'
#' @section Input and Output Channels: `r roxy_pipeop_torch_channels_default()`
#' @section State: `r roxy_pipeop_torch_state_default()`
#'
#' @section Parameters:
#' No parameters.
#'
#' @section Fields: `r roxy_pipeop_torch_fields_default()`
#' @section Methods: `r roxy_pipeop_torch_methods_default()`
#' @section Internals: Calls [`torch::nn_sigmoid()`] when trained.
#' @section Credit: `r roxy_pipeop_torch_license()`
#' @family PipeOpTorch
#' @export
PipeOpTorchSigmoid = make_activation("sigmoid", "Sigmoid", param_set = ps())

#' @title SoftPlus Activation Function
#'
#' @usage NULL
#' @name mlr_pipeops_torch_softplus
#' @format `r roxy_pipeop_torch_format()`
#'
#' @inherit torch::nnf_softplus description
#'
#' @section Construction: `r roxy_pipeop_torch_construction("SoftPlus")`
#' `r roxy_param_id("nn_softplus")`
#' `r roxy_param_param_vals()`
#'
#' @section Input and Output Channels: `r roxy_pipeop_torch_channels_default()`
#' @section State: `r roxy_pipeop_torch_state_default()`
#'
#' @section Parameters:
#' * `beta` :: `numeric(1)`\cr
#'   The beta value for the Softplus formulation. Default: 1
#' * `threshold` :: `numeric(1)`\cr
#'   Values above this revert to a linear function. Default: 20
#'
#' @section Fields: `r roxy_pipeop_torch_fields_default()`
#' @section Methods: `r roxy_pipeop_torch_methods_default()`
#' @section Internals: Calls [`torch::nn_softplus()`] when trained.
#' @section Credit: `r roxy_pipeop_torch_license()`
#' @family PipeOpTorch
#' @export
PipeOpTorchSoftPlus = make_activation("softplus", "SoftPlus",
  param_set = ps(
  beta = p_dbl(default = 1, tags = "train"),
  threshold = p_dbl(default = 20, tags = "train")
))

#' @title Soft Shrink Activation Function
#'
#' @usage NULL
#' @name mlr_pipeops_torch_softshrink
#' @format `r roxy_pipeop_torch_format()`
#'
#' @inherit torch::nnf_softshrink description
#'
#' @section Construction: `r roxy_pipeop_torch_construction("SoftShrink")`
#' `r roxy_param_id("nn_softshrink")`
#' `r roxy_param_param_vals()`
#'
#' @section Input and Output Channels: `r roxy_pipeop_torch_channels_default()`
#' @section State: `r roxy_pipeop_torch_state_default()`
#'
#' @section Parameters:
#' * `lamd` :: `numeric(1)`\cr
#'   The lambda (must be no less than zero) value for the Softshrink formulation. Default: 0.5
#'
#' @section Fields: `r roxy_pipeop_torch_fields_default()`
#' @section Methods: `r roxy_pipeop_torch_methods_default()`
#' @section Internals: Calls [`torch::nn_softshrink()`] when trained.
#' @section Credit: `r roxy_pipeop_torch_license()`
#' @family PipeOpTorch
#' @export
PipeOpTorchtSoftShrink = make_activation("softshrink", "SoftShrink", param_set = ps(
  lambd = p_dbl(default = 0.5, upper = 1, tags = "train")
))

#' @title SoftSign Activation Function
#'
#' @usage NULL
#' @name mlr_pipeops_torch_softsign
#' @format `r roxy_pipeop_torch_format()`
#'
#' @inherit torch::nnf_softsign description
#'
#' @section Construction: `r roxy_pipeop_torch_construction("SoftSign")`
#' `r roxy_param_id("nn_softsign")`
#' `r roxy_param_param_vals()`
#'
#' @section Input and Output Channels: `r roxy_pipeop_torch_channels_default()`
#' @section State: `r roxy_pipeop_torch_state_default()`
#'
#' @section Parameters:
#' No parameters.
#'
#' @section Fields: `r roxy_pipeop_torch_fields_default()`
#' @section Methods: `r roxy_pipeop_torch_methods_default()`
#' @section Internals: Calls [`torch::nn_softsign()`] when trained.
#' @section Credit: `r roxy_pipeop_torch_license()`
#' @family PipeOpTorch
#' @export
PipeOpTorchActSoftSign = make_activation("softsign", "SoftSign", param_set = ps())

#' @title Tanh Activation Function
#'
#' @usage NULL
#' @name mlr_pipeops_torch_tanh
#' @format `r roxy_pipeop_torch_format()`
#'
#' @inherit torch::nn_tanh description
#'
#' @section Construction: `r roxy_pipeop_torch_construction("Tanh")`
#' `r roxy_param_id("nn_tanh")`
#' `r roxy_param_param_vals()`
#'
#' @section Input and Output Channels: `r roxy_pipeop_torch_channels_default()`
#' @section State: `r roxy_pipeop_torch_state_default()`
#'
#' @section Parameters:
#' No parameters.
#'
#' @section Fields: `r roxy_pipeop_torch_fields_default()`
#' @section Methods: `r roxy_pipeop_torch_methods_default()`
#' @section Internals: Calls [`torch::nn_tanh()`] when trained.
#' @section Credit: `r roxy_pipeop_torch_license()`
#' @family PipeOpTorch
#' @export
PipeOpTorchActTanh = make_activation("tanh", "Tanh", param_set = ps())

#' @title Tanh Shrink Activation Function
#'
#' @usage NULL
#' @name mlr_pipeops_torch_tanhshrink
#' @format `r roxy_pipeop_torch_format()`
#'
#' @inherit torch::nnf_tanhshrink description
#'
#' @section Construction: `r roxy_pipeop_torch_construction("TanhShrink")`
#' `r roxy_param_id("nn_tanhshrink")`
#' `r roxy_param_param_vals()`
#'
#' @section Input and Output Channels: `r roxy_pipeop_torch_channels_default()`
#' @section State: `r roxy_pipeop_torch_state_default()`
#'
#' @section Parameters:
#' No parameters.
#'
#' @section Fields: `r roxy_pipeop_torch_fields_default()`
#' @section Methods: `r roxy_pipeop_torch_methods_default()`
#' @section Internals: Calls [`torch::nn_tanhshrink()`] when trained.
#' @section Credit: `r roxy_pipeop_torch_license()`
#' @family PipeOpTorch
#' @export
PipeOpTorchActTanhShrink = make_activation("tanhshrink", "TanhShrink", param_set = ps())

#' @title Treshol Activation Function
#'
#' @usage NULL
#' @name mlr_pipeops_torch_threshold
#' @format `r roxy_pipeop_torch_format()`
#'
#' @inherit torch::nnf_threshold description
#'
#' @section Construction: `r roxy_pipeop_torch_construction("Threshold")`
#' `r roxy_param_id("nn_threshold")`
#' `r roxy_param_param_vals()`
#'
#' @section Input and Output Channels: `r roxy_pipeop_torch_channels_default()`
#' @section State: `r roxy_pipeop_torch_state_default()`
#'
#' @section Parameters:
#' * `threshold` :: `numeric(1)`\cr
#'   The value to threshold at.
#' * `value` :: `numeric(1)`\cr
#'   The value to replace with.
#' * `inplace` :: `logical(1)`\cr
#'   Can optionally do the operation in-place. Default: ‘FALSE’.
#'
#' @section Fields: `r roxy_pipeop_torch_fields_default()`
#' @section Methods: `r roxy_pipeop_torch_methods_default()`
#' @section Internals: Calls [`torch::nn_threshold()`] when trained.
#' @section Credit: `r roxy_pipeop_torch_license()`
#' @family PipeOpTorch
#' @export
PipeOpTorchThreshold = make_activation("threshold", "Threshold", param_set = ps(
  threshold = p_dbl(tags = "train"),
  value = p_dbl(tags = "train"),
  inplace = p_lgl(default = FALSE, tags = "train")
))

#' @title GLU Activation Function
#'
#' @usage NULL
#' @name mlr_pipeops_torch_glu
#' @format `r roxy_pipeop_torch_format()`
#'
#' @inherit torch::nnf_glu description
#'
#' @section Construction: `r roxy_pipeop_torch_construction("GLU")`
#' `r roxy_param_id("nn_glu")`
#' `r roxy_param_param_vals()`
#'
#' @section Input and Output Channels: `r roxy_pipeop_torch_channels_default()`
#' @section State: `r roxy_pipeop_torch_state_default()`
#'
#' @section Parameters:
#' * `dim` :: `integer(1)`\cr
#'   Dimension on which to split the input. Default: -1
#'
#' @section Fields: `r roxy_pipeop_torch_fields_default()`
#' @section Methods: `r roxy_pipeop_torch_methods_default()`
#' @section Internals: Calls [`torch::nn_glu()`] when trained.
#' @section Credit: `r roxy_pipeop_torch_license()`
#' @family PipeOpTorch
#' @export
PipeOpTorchGLU = R6Class("PipeOpTorchGLU",
  inherit = PipeOpTorch,
  public = list(
    initialize = function(id = "glu", param_vals = list()) {
      param_set = ps(
        dim = p_int(default = -1L, lower = 1L, tags = "train", special_vals = list(-1L))
      )
      super$initialize(
      id = id,
      param_set = param_set,
      param_vals = param_vals,
      module_generator = nn_glu,
      tags = "activation"
      )
    }
  ),
  private = list(
    .shapes_out = function(shapes_in, param_vals) {
      dim = param_vals$dim
      assert_true(dim %% 2 == 0)
      if (dim == -1) {
        shapes_in[[1L]][length(shapes_in[[1L]])] = shapes_in[[1L]][length(shapes_in[[1L]])] / 2
      }
      shapes_in
    }
  )
)
