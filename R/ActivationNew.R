#' @include zzz.R
make_activation = function(name, param_set, parent_env = parent.frame()) {
  classname = paste0("PipeOpTorchAct", capitalize(name))
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

#' @title Elu Activation Function
#'
#' @usage NULL
#' @name pipeop_torch_elu
#' @format `r roxy_pipeop_torch_format()`
#'
#' @inherit torch::nnf_elu description
#'
#' @section Construction: `r roxy_pipeop_torch_construction("PipeOpTorchAct")`
#' `r roxy_param_id("elu")`
#' `r roxy_param_param_vals()`
#'
#' @section Input and Output Channels:
#' `r roxy_pipeop_torch_channels_default()`
#'
#' @section State:
#' `r roxy_pipeop_torch_state_default()`
#'
#' @section Parameters:
#' * `alpha` :: `numeric(1)`\cr
#'   The alpha value for the ELU formulation. Default: 1.0
#' * `inplace` :: `logical(1)`\cr
#'   Whether to do the operation in-place. Default: `FALSE`.
#'
#' @section Internals:
#' Calls [`torch::nn_elu()`] when trained.
#'
#' @section Credit:
#' `r roxy_param_torch_license()`
#'
#' @family PipeOpTorch
#' @export
PipeOpTorchElu = make_activation("elu", param_set = ps(
  alpha = p_dbl(default = 1, tags = "train"),
  inplace = p_lgl(default = FALSE, tags = "train")
))

#' @title Hard Shrink Activation Function
#'
#' @usage NULL
#' @name pipeop_torch_hard_shrink
#' @format `r roxy_pipeop_torch_format()`
#'
#' @inherit torch::nnf_hardshrink description
#'
#' @section Construction: `r roxy_pipeop_torch_construction("HardShrink")`
#' `r roxy_param_id("elu")`
#' `r roxy_param_param_vals()`
#'
#' @section Input and Output Channels:
#' `r roxy_pipeop_torch_channels_default()`
#'
#' @section State:
#' `r roxy_pipeop_torch_state_default()`
#'
#' @section Parameters:
#' * `alpha` :: `numeric(1)`\cr
#'   The alpha value for the ELU formulation. Default: 1.0
#' * `inplace` :: `logical(1)`\cr
#'   Whether to do the operation in-place. Default: `FALSE`.
#'
#' @section Internals:
#' Calls [`torch::nn_hardshrink()`] when trained.
#'
#' @section Credit:
#' `r roxy_param_torch_license()`
#'
#' @family PipeOpTorch
#' @export
PipeOpTorchHardShrink = make_activation("hardshrink", param_set = ps(
  lambd = p_dbl(default = 0.5, tags = "train")
))

#' @export
PipeOpTorchHardSigmoid = make_activation("hardsigmoid", param_set = ps())

#' @export
PipeOpTorchHardTanh = make_activation("hardtanh", param_set = ps(
  min_val = p_dbl(default = -1, tags = "train"),
  max_val = p_dbl(default = 1, tags = "train"),
  inplace = p_lgl(default = FALSE, tags = "train")
))

#' @export
PipeOpTorchHardSwish = make_activation("hardswish", param_set = ps())

#' @export
PipeOpTorchLeakyRelu = make_activation("leaky_relu", param_set = ps(
  negative_slope = p_dbl(default = 0.01, tags = "train"),
  inplace = p_lgl(default = FALSE, tags = "train")
))

#' @export
PipeOpTorchLogSigmoid = make_activation("log_sigmoid", param_set = ps())

#' @export
PipeOpTorchPrelu = make_activation("prelu", param_set = ps(
  num_parameters = p_int(1, tags = "train"),
  init = p_dbl(default = 0.25, tags = "train")
))

#' @export
PipeOpTorchRelu = make_activation("relu", param_set = ps(
  inplace = p_lgl(default = FALSE, tags = "train")
))

#' @export
PipeOpTorchRelu6 = make_activation("relu6", param_set = ps(
  inplace = p_lgl(default = FALSE, tags = "train")
))

#' @export
PipeOpTorchRrelu = make_activation("rrelu", param_set = ps(
  lower = p_dbl(default = 1 / 8, tags = "train"),
  upper = p_dbl(default = 1 / 3, tags = "train"),
  inplace = p_lgl(default = FALSE, tags = "train")
))

#' @export
PipeOpTorchSelu = make_activation("selu", param_set = ps(
  inplace = p_lgl(tags = "train")
))

#' @export
PipeOpTorchSelu = make_activation("celu", param_set = ps(
  alpha = p_dbl(default = 1.0, tags = "train"),
  inplace = p_lgl(default = FALSE, tags = "train")
))

#' @export
PipeOpTorchSigmoid = make_activation("gelu", param_set = ps())

#' @export
PipeOpTorchSigmoid = make_activation("sigmoid", param_set = ps())

#' @export
PipeOpTorchSoftPlus = make_activation("softplus", param_set = ps(
  beta = p_dbl(default = 1, tags = "train"),
  threshold = p_dbl(default = 20, tags = "train")
))

#' @export
PipeOpTorchtSoftShrink = make_activation("softshrink", param_set = ps(
  lambd = p_dbl(default = 0.5, upper = 1, tags = "train")
))

#' @export
PipeOpTorchActSoftSign = make_activation("softsign", param_set = ps())

#' @export
PipeOpTorchActTanh = make_activation("tanh", param_set = ps())

#' @export
PipeOpTorchActTanhShrink = make_activation("tanhshrink", param_set = ps())

#' @export
PipeOpTorchActThreshold = make_activation("threshold", param_set = ps(
  threshold = p_dbl(tags = "train"),
  value = p_dbl(tags = "train"),
  inplace = p_lgl(default = FALSE, tags = "train")
))

#' @export
PipeOpTorchActGlu = make_activation("glu", param_set = ps(
  dim = p_int(default = -1L, lower = 1L, tags = "train", special_vals = list(-1L))
))
