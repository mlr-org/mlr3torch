#' @title ELU Activation Function
#'
#' @inherit torch::nnf_elu description
#' @section nn_module: Calls [`torch::nn_elu()`] when trained.
#' @section Parameters:
#' * `alpha` :: `numeric(1)`\cr
#'   The alpha value for the ELU formulation. Default: 1.0
#' * `inplace` :: `logical(1)`\cr
#'   Whether to do the operation in-place. Default: `FALSE`.
#'
#' @templateVar id nn_elu
#' @template pipeop_torch_channels_default
#' @template pipeop_torch
#' @template pipeop_torch_example
#' @export
PipeOpTorchELU = R6Class("PipeOpTorchELU",
  inherit = PipeOpTorch,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    initialize = function(id = "nn_elu", param_vals = list()) {
      param_set = ps(
        alpha   = p_dbl(default = 1, tags = "train"),
        inplace = p_lgl(default = FALSE, tags = "train")
      )
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals,
        module_generator = nn_elu,
        tags = "activation"
      )
    }
  )
)

register_po("nn_elu", PipeOpTorchELU)

#' @title Hard Shrink Activation Function
#' @inherit torch::nnf_hardshrink description
#' @section nn_module: Calls [`torch::nn_hardshrink()`] when trained.
#' @section Parameters:
#' * `lambd` :: `numeric(1)`\cr
#'   The lambda value for the Hardshrink formulation formulation. Default 0.5.
#' @templateVar id nn_hardshrink
#' @template pipeop_torch_channels_default
#' @template pipeop_torch
#' @template pipeop_torch_example
#' @export
PipeOpTorchHardShrink = R6Class("PipeOpTorchHardShrink",
  inherit = PipeOpTorch,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    initialize = function(id = "nn_hardshrink", param_vals = list()) {
      param_set = ps(
        lambd = p_dbl(default = 0.5, tags = "train")
      )
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals,
        module_generator = nn_hardshrink,
        tags = "activation"
      )
    }
  )
)

register_po("nn_hardshrink", PipeOpTorchHardShrink)

#' @title Hard Sigmoid Activation Function
#'
#' @inherit torch::nnf_hardsigmoid description
#' @section nn_module: Calls [`torch::nn_hardsigmoid()`] when trained.
#' @section Parameters:
#' No parameters.
#' @templateVar id nn_hardsigmoid
#' @template pipeop_torch_channels_default
#' @template pipeop_torch
#' @template pipeop_torch_example
#'
#' @export
PipeOpTorchHardSigmoid = R6Class("PipeOpTorchHardSigmoid",
  inherit = PipeOpTorch,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    initialize = function(id = "nn_hardsigmoid", param_vals = list()) {
      param_set = ps()
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals,
        module_generator = nn_hardsigmoid,
        tags = "activation"
      )
    }
  )
)

register_po("nn_hardsigmoid", PipeOpTorchHardSigmoid)

#' @title Hard Tanh Activation Function
#'
#' @inherit torch::nnf_hardtanh description
#' @section nn_module: Calls [`torch::nn_hardtanh()`] when trained.
#' @section Parameters:
#' * `min_val` :: `numeric(1)`\cr
#'   Minimum value of the linear region range. Default: -1.
#' * `max_val` :: `numeric(1)`\cr
#'   Maximum value of the linear region range. Default: 1.
#' * `inplace` :: `logical(1)`\cr
#'   Can optionally do the operation in-place. Default: `FALSE`.
#' @templateVar id nn_hardtanh
#'
#' @template pipeop_torch
#' @template pipeop_torch_example
#'
#'
#' @export
PipeOpTorchHardTanh = R6Class("PipeOpTorchHardTanh",
  inherit = PipeOpTorch,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    initialize = function(id = "nn_hardtanh", param_vals = list()) {
      param_set = ps(
        min_val = p_dbl(default = -1, tags = "train"),
        max_val = p_dbl(default = 1, tags = "train"),
        inplace = p_lgl(default = FALSE, tags = "train")
      )
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals,
        module_generator = nn_hardtanh,
        tags = "activation"
      )
    }
  )
)

register_po("nn_hardtanh", PipeOpTorchHardTanh)


#' @title Leaky ReLU Activation Function
#'
#' @inherit torch::nnf_leaky_relu description
#' @section nn_module: Calls [`torch::nn_leaky_relu()`] when trained.
#' @section Parameters:
#' * `negative_slope` :: `numeric(1)`\cr
#'   Controls the angle of the negative slope. Default: 1e-2.
#' * `inplace` :: `logical(1)`\cr
#'   Can optionally do the operation in-place. Default: ‘FALSE’.
#' @templateVar id nn_leaky_relu
#' @template pipeop_torch_channels_default
#' @template pipeop_torch
#' @template pipeop_torch_example
#'
#' @export
PipeOpTorchLeakyReLU = R6Class("PipeOpTorchLeakyReLU",
  inherit = PipeOpTorch,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    initialize = function(id = "nn_leaky_relu", param_vals = list()) {
      param_set = ps(
        negative_slope = p_dbl(default = 0.01, tags = "train"),
        inplace        = p_lgl(default = FALSE, tags = "train")
      )
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals,
        module_generator = nn_leaky_relu,
        tags = "activation"
      )
    }
  )
)

register_po("nn_leaky_relu", PipeOpTorchLeakyReLU)

#' @title Log Sigmoid Activation Function
#'
#' @inherit torch::nnf_logsigmoid description
#' @section nn_module: Calls [`torch::nn_log_sigmoid()`] when trained.
#' @section Parameters:
#' No parameters.
#'
#' @templateVar id nn_log_sigmoid
#' @template pipeop_torch_channels_default
#' @template pipeop_torch
#' @template pipeop_torch_example
#'
#'
#' @export
PipeOpTorchLogSigmoid = R6Class("PipeOpTorchLogSigmoid",
  inherit = PipeOpTorch,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    initialize = function(id = "nn_log_sigmoid", param_vals = list()) {
      param_set = ps()
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals,
        module_generator = nn_log_sigmoid,
        tags = "activation"
      )
    }
  )
)

register_po("nn_log_sigmoid", PipeOpTorchLogSigmoid)

#' @title PReLU Activation Function
#'
#' @inherit torch::nnf_prelu description
#' @section nn_module: Calls [`torch::nn_prelu()`] when trained.
#' @section Parameters:
#' * `num_parameters` :: `integer(1)`:
#'   Number of a to learn. Although it takes an int as input, there is only two values are legitimate: 1, or the
#'   number of channels at input. Default: 1.
#' * `init` :: `numeric(1)`\cr T
#'   The initial value of a. Default: 0.25.
#' @templateVar id nn_prelu
#' @template pipeop_torch_channels_default
#' @template pipeop_torch
#' @template pipeop_torch_example
#'
#' @export
PipeOpTorchPReLU = R6Class("PipeOpTorchPReLU",
  inherit = PipeOpTorch,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    initialize = function(id = "nn_prelu", param_vals = list()) {
      param_set = ps(
        num_parameters = p_int(default = 1, lower = 1, tags = "train"),
        init           = p_dbl(default = 0.25, tags = "train")
      )
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals,
        module_generator = nn_prelu,
        tags = "activation"
      )
    }
  )
)

register_po("nn_prelu", PipeOpTorchPReLU)

#' @title ReLU Activation Function
#'
#' @inherit torch::nnf_relu description
#' @section nn_module: Calls [`torch::nn_relu()`] when trained.
#' @section Parameters:
#' * `inplace` :: `logical(1)`\cr
#'   Whether to do the operation in-place. Default: `FALSE`.
#' @templateVar id nn_relu
#' @template pipeop_torch_channels_default
#' @template pipeop_torch
#' @template pipeop_torch_example
#'
#'
#' @export
PipeOpTorchReLU = R6Class("PipeOpTorchReLU",
  inherit = PipeOpTorch,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    initialize = function(id = "nn_relu", param_vals = list()) {
      param_set = ps(
        inplace = p_lgl(default = FALSE, tags = "train")
      )
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals,
        module_generator = nn_relu,
        tags = "activation"
      )
    }
  )
)

register_po("nn_relu", PipeOpTorchReLU)

#' @title ReLU6 Activation Function
#'
#' @inherit torch::nnf_relu6 description
#' @section nn_module: Calls [`torch::nn_relu6()`] when trained.
#' @section Parameters:
#' * `inplace` :: `logical(1)`\cr
#'   Whether to do the operation in-place. Default: `FALSE`.
#' @templateVar id nn_relu6
#' @template pipeop_torch_channels_default
#' @template pipeop_torch
#' @template pipeop_torch_example
#'
#' @export
PipeOpTorchReLU6 = R6Class("PipeOpTorchReLU6",
  inherit = PipeOpTorch,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    initialize = function(id = "nn_relu6", param_vals = list()) {
      param_set = ps(
        inplace = p_lgl(default = FALSE, tags = "train")
      )
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals,
        module_generator = nn_relu6,
        tags = "activation"
      )
    }
  )
)

register_po("nn_relu6", PipeOpTorchReLU6)

#' @title RReLU Activation Function
#' @inherit torch::nnf_rrelu description
#' @section nn_module: Calls [`torch::nn_rrelu()`] when trained.
#' @section Parameters:
#' * `lower`:: `numeric(1)`\cr
#'   Lower bound of the uniform distribution. Default: 1/8.
#' * `upper`:: `numeric(1)`\cr
#'   Upper bound of the uniform distribution. Default: 1/3.
#' * `inplace` :: `logical(1)`\cr
#'   Whether to do the operation in-place. Default: `FALSE`.
#'
#' @templateVar id nn_rrelu
#' @template pipeop_torch_channels_default
#' @template pipeop_torch
#' @template pipeop_torch_example
#'
#'
#' @export
PipeOpTorchRReLU = R6Class("PipeOpTorchRReLU",
  inherit = PipeOpTorch,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    initialize = function(id = "nn_rrelu", param_vals = list()) {
      param_set = ps(
        lower = p_dbl(default = 1 / 8, tags = "train"),
        upper = p_dbl(default = 1 / 3, tags = "train"),
        inplace = p_lgl(default = FALSE, tags = "train")
      )
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals,
        module_generator = nn_rrelu,
        tags = "activation"
      )
    }
  )
)

register_po("nn_rrelu", PipeOpTorchRReLU)

#' @title SELU Activation Function
#'
#' @inherit torch::nnf_selu description
#' @section nn_module: Calls [`torch::nn_selu()`] when trained.
#' @section Parameters:
#' * `inplace` :: `logical(1)`\cr
#'   Whether to do the operation in-place. Default: `FALSE`.
#' @templateVar id nn_selu
#' @template pipeop_torch_channels_default
#' @template pipeop_torch
#' @template pipeop_torch_example
#'
#'
#' @export
PipeOpTorchSELU = R6Class("PipeOpTorchSELU",
  inherit = PipeOpTorch,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    initialize = function(id = "nn_selu", param_vals = list()) {
      param_set = ps(
        inplace = p_lgl(default = FALSE, tags = "train")
      )
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals,
        module_generator = nn_selu,
        tags = "activation"
      )
    }
  )
)

register_po("nn_selu", PipeOpTorchSELU)

#' @title CELU Activation Function
#'
#' @inherit torch::nnf_celu description
#' @section nn_module: Calls [`torch::nn_celu()`] when trained.
#' @section Parameters:
#' * `alpha` :: `numeric(1)`\cr
#'   The alpha value for the ELU formulation. Default: 1.0
#' * `inplace` :: `logical(1)`\cr
#'   Whether to do the operation in-place. Default: `FALSE`.
#' @templateVar id nn_celu
#' @template pipeop_torch_channels_default
#' @template pipeop_torch
#' @template pipeop_torch_example
#'
#'
#' @export
PipeOpTorchCELU = R6Class("PipeOpTorchCELU",
  inherit = PipeOpTorch,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    initialize = function(id = "nn_celu", param_vals = list()) {
      param_set = ps(
        alpha   = p_dbl(default = 1.0, tags = "train"),
        inplace = p_lgl(default = FALSE, tags = "train")
      )
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals,
        module_generator = nn_celu,
        tags = "activation"
      )
    }
  )
)

register_po("nn_celu", PipeOpTorchCELU)

#' @title GELU Activation Function
#'
#' @inherit torch::nnf_gelu description
#' @section nn_module: Calls [`torch::nn_gelu()`] when trained.
#' @section Parameters:
#' * `approximate` :: `character(1)`\cr
#'   Whether to use an approximation algorithm. Default is `"none"`.
#' @templateVar id nn_gelu
#' @template pipeop_torch_channels_default
#' @template pipeop_torch
#' @template pipeop_torch_example
#'
#'
#' @export
PipeOpTorchGELU = R6Class("PipeOpTorchGELU",
  inherit = PipeOpTorch,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    initialize = function(id = "nn_gelu", param_vals = list()) {
      param_set = ps(
        approximate = p_fct(default = "none", levels = c("none", "tanh"), tags = "train")
      )
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals,
        module_generator = nn_gelu,
        tags = "activation"
      )
    }
  )
)

register_po("nn_gelu", PipeOpTorchGELU)

#' @title Sigmoid Activation Function
#'
#' @inherit torch::nnf_sigmoid description
#' @section nn_module: Calls [`torch::nn_sigmoid()`] when trained.
#' @section Parameters:
#' No parameters.
#'
#' @templateVar id nn_sigmoid
#' @template pipeop_torch_channels_default
#' @template pipeop_torch
#' @template pipeop_torch_example
#'
#' @export
PipeOpTorchSigmoid = R6Class("PipeOpTorchSigmoid",
  inherit = PipeOpTorch,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    initialize = function(id = "nn_sigmoid", param_vals = list()) {
      param_set = ps()
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals,
        module_generator = nn_sigmoid,
        tags = "activation"
      )
    }
  )
)

register_po("nn_sigmoid", PipeOpTorchSigmoid)

#' @title SoftPlus Activation Function
#'
#' @inherit torch::nnf_softplus description
#' @section nn_module: Calls [`torch::nn_softplus()`] when trained.
#' @section Parameters:
#' * `beta` :: `numeric(1)`\cr
#'   The beta value for the Softplus formulation. Default: 1
#' * `threshold` :: `numeric(1)`\cr
#'   Values above this revert to a linear function. Default: 20
#' @templateVar id nn_softplus
#' @template pipeop_torch_channels_default
#' @template pipeop_torch
#' @template pipeop_torch_example
#'
#' @export
PipeOpTorchSoftPlus = R6Class("PipeOpTorchSoftPlus",
  inherit = PipeOpTorch,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    initialize = function(id = "nn_softplus", param_vals = list()) {
      param_set = ps(
        beta = p_dbl(default = 1, tags = "train"),
        threshold = p_dbl(default = 20, tags = "train")
      )
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals,
        module_generator = nn_softplus,
        tags = "activation"
      )
    }
  )
)


register_po("nn_softplus", PipeOpTorchSoftPlus)

#' @title Soft Shrink Activation Function
#'
#' @inherit torch::nnf_softshrink description
#' @section nn_module: Calls [`torch::nn_softshrink()`] when trained.
#' @section Parameters:
#' * `lamd` :: `numeric(1)`\cr
#'   The lambda (must be no less than zero) value for the Softshrink formulation. Default: 0.5
#' @templateVar id nn_softshrink
#' @template pipeop_torch_channels_default
#' @template pipeop_torch
#' @template pipeop_torch_example
#'
#' @export
PipeOpTorchSoftShrink = R6Class("PipeOpTorchSoftShrink",
  inherit = PipeOpTorch,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    initialize = function(id = "nn_softshrink", param_vals = list()) {
      param_set = ps(
        lambd = p_dbl(default = 0.5, upper = 1, tags = "train")
      )
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals,
        module_generator = nn_softshrink,
        tags = "activation"
      )
    }
  )
)

register_po("nn_softshrink", PipeOpTorchSoftShrink)

#' @title SoftSign Activation Function
#'
#' @inherit torch::nnf_softsign description
#' @section nn_module: Calls [`torch::nn_softsign()`] when trained.
#' @section Parameters:
#' No parameters.
#' @templateVar id nn_softsign
#' @template pipeop_torch_channels_default
#' @template pipeop_torch
#' @template pipeop_torch_example
#'
#' @export
PipeOpTorchSoftSign = R6Class("PipeOpTorchSoftSign",
  inherit = PipeOpTorch,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    initialize = function(id = "nn_softsign", param_vals = list()) {
      param_set = ps()
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals,
        module_generator = nn_softsign,
        tags = "activation"
      )
    }
  )
)

register_po("nn_softsign", PipeOpTorchSoftSign)

#' @title Tanh Activation Function
#'
#' @inherit torch::nn_tanh description
#' @section nn_module: Calls [`torch::nn_tanh()`] when trained.
#' @section Parameters:
#' No parameters.
#' @templateVar id nn_tanh
#' @template pipeop_torch_channels_default
#' @template pipeop_torch
#' @template pipeop_torch_example
#'
#'
#' @export
PipeOpTorchTanh = R6Class("PipeOpTorchTanh",
  inherit = PipeOpTorch,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    initialize = function(id = "nn_tanh", param_vals = list()) {
      param_set = ps()
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals,
        module_generator = nn_tanh,
        tags = "activation"
      )
    }
  )
)

register_po("nn_tanh", PipeOpTorchTanh)

#' @title Tanh Shrink Activation Function
#'
#' @inherit torch::nnf_tanhshrink description
#' @section nn_module: Calls [`torch::nn_tanhshrink()`] when trained.
#' @section Parameters:
#' No parameters.
#' @templateVar id nn_tanhshrink
#' @template pipeop_torch_channels_default
#' @template pipeop_torch
#' @template pipeop_torch_example
#'
#'
#' @export
PipeOpTorchTanhShrink = R6Class("PipeOpTorchTanhShrink",
  inherit = PipeOpTorch,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    initialize = function(id = "nn_tanhshrink", param_vals = list()) {
      param_set = ps()
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals,
        module_generator = nn_tanhshrink,
        tags = "activation"
      )
    }
  )
)

register_po("nn_tanhshrink", PipeOpTorchTanhShrink)

#' @title Treshold Activation Function
#'
#' @inherit torch::nnf_threshold description
#' @section nn_module: Calls [`torch::nn_threshold()`] when trained.
#' @section Parameters:
#' * `threshold` :: `numeric(1)`\cr
#'   The value to threshold at.
#' * `value` :: `numeric(1)`\cr
#'   The value to replace with.
#' * `inplace` :: `logical(1)`\cr
#'   Can optionally do the operation in-place. Default: ‘FALSE’.
#' @templateVar id nn_threshold
#' @template pipeop_torch_channels_default
#' @templateVar param_vals threshold = 1, value = 2
#' @template pipeop_torch
#' @template pipeop_torch_example
#'
#'
#' @export
PipeOpTorchThreshold = R6Class("PipeOpTorchThreshold",
  inherit = PipeOpTorch,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    initialize = function(id = "nn_threshold", param_vals = list()) {
      param_set = ps(
        threshold = p_dbl(tags = c("train", "required")),
        value     = p_dbl(tags = c("train", "required")),
        inplace   = p_lgl(default = FALSE, tags = "train")
      )
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals,
        module_generator = nn_threshold,
        tags = "activation"
      )
    }
  )
)

register_po("nn_threshold", PipeOpTorchThreshold)

#' @title GLU Activation Function
#'
#' @inherit torch::nnf_glu description
#' @section nn_module: Calls [`torch::nn_glu()`] when trained.
#' @section Parameters:
#' * `dim` :: `integer(1)`\cr
#'   Dimension on which to split the input. Default: -1
#' @templateVar id nn_glu
#' @template pipeop_torch_channels_default
#' @template pipeop_torch
#' @template pipeop_torch_example
#'
#'
#' @export
PipeOpTorchGLU = R6Class("PipeOpTorchGLU",
  inherit = PipeOpTorch,
  public = list(
    #' @description Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    initialize = function(id = "nn_glu", param_vals = list()) {
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
    .shapes_out = function(shapes_in, param_vals, task) {
      shape = shapes_in[[1L]]
      true_dim = param_vals$dim %??% -1
      if (true_dim < 0) {
        true_dim = 1 + length(shape) + true_dim
      }
      d_new = shape[true_dim] / 2
      if (test_integerish(d_new)) {
        shape[true_dim] = d_new
      } else {
        stopf("Dimension %i of input tensor must be divisible by 2.", true_dim)
      }
      list(shape)
    }
  )
)

register_po("nn_glu", PipeOpTorchGLU)

reglu = function(x) {
  assert_true(last(x$shape, 1) %% 2 == 0)
  chunked = x$chunk(2, dim = -1)
  a = chunked[[1]]
  b = chunked[[2]]
  return(a * nnf_relu(b))
}

#' @title ReGLU Module
#'
#' @description
#' Rectified Gated Linear Unit (ReGLU) module.
#' Computes the output as \eqn{\text{ReGLU}(x, g) = x \cdot \text{ReLU}(g)}
#' where \(x\) and \(g\) are created by splitting the input tensor in half along the last dimension.
#'
#' @export
#' @references
#' `r format_bib("shazeer2020glu")`
#' @examplesIf torch::torch_is_installed()
#' x = torch::torch_randn(10, 10)
#' reglu = nn_reglu()
#' reglu(x)
nn_reglu = nn_module(
  "nn_reglu",
  forward = function(input) {
    return(reglu(input))
  }
)

#' @title ReGLU Activation Function
#'
#' @description
#' Rectified Gated Linear Unit (ReGLU) activation function.
#' See [`nn_reglu`] for details.
#' @section Parameters:
#' No parameters.
#' @templateVar id nn_reglu
#' @template pipeop_torch_channels_default
#' @template pipeop_torch
#' @template pipeop_torch_example
#' @export
PipeOpTorchReGLU = R6Class("PipeOpTorchReGLU",
  inherit = PipeOpTorch,
  public = list(
    #' @description Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    initialize = function(id = "nn_reglu", param_vals = list()) {
      param_set = ps()
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals,
        module_generator = nn_reglu,
        tags = "activation"
      )
    }
  ),
  private = list(
    .shapes_out = function(shapes_in, param_vals, task) {
      shape = shapes_in[[1L]]
      d_new = last(shape, 1) / 2
      if (test_integerish(d_new)) {
        shape[length(shape)] = d_new
        list(shape)
      } else {
        stopf("Last dimension of input tensor must be divisible by 2.")
      }
    }
  )
)

register_po("nn_reglu", PipeOpTorchReGLU)

geglu = function(x) {
  assert_true(last(x$shape, 1) %% 2 == 0)
  chunked = x$chunk(2, dim = -1)
  a = chunked[[1]]
  b = chunked[[2]]
  return(a * nnf_gelu(b))
}

#' @title GeGLU Module
#' @description
#' This module implements the Gaussian Error Linear Unit Gated Linear Unit (GeGLU) activation function.
#' It computes \eqn{\text{GeGLU}(x, g) = x \cdot \text{GELU}(g)}
#' where \(x\) and \(g\) are created by splitting the input tensor in half along the last dimension.
#'
#' @export
#' @references
#' `r format_bib("shazeer2020glu")`
#' @examplesIf torch::torch_is_installed()
#' x = torch::torch_randn(10, 10)
#' glu = nn_geglu()
#' glu(x)
nn_geglu = nn_module(
  "nn_geglu",
  forward = function(input) {
    return(geglu(input))
  }
)

#' @title GeGLU Activation Function
#'
#' @description
#' Gaussian Error Linear Unit Gated Linear Unit (GeGLU) activation function, see
#' [`nn_geglu`] for details.
#' @section Parameters:
#' No parameters.
#' @templateVar id nn_geglu
#' @template pipeop_torch_channels_default
#' @template pipeop_torch
#' @template pipeop_torch_example
#'
#' @export
PipeOpTorchGeGLU = R6Class("PipeOpTorchGeGLU",
  inherit = PipeOpTorch,
  public = list(
    #' @description Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    initialize = function(id = "nn_geglu", param_vals = list()) {
      param_set = ps()
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals,
        module_generator = nn_geglu,
        tags = "activation"
      )
    }
  ),
  private = list(
    .shapes_out = function(shapes_in, param_vals, task) {
      shape = shapes_in[[1L]]
      d_new = last(shape, 1) / 2
      if (test_integerish(d_new)) {
        shape[length(shape)] = d_new
        list(shape)
      } else {
        stopf("Last dimension of input tensor must be divisible by 2.")
      }
    }
  )
)

register_po("nn_geglu", PipeOpTorchGeGLU)
