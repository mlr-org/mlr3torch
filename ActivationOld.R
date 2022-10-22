#' @title Activation Function
#'
#' @usage NULL
#' @name pipeop_torch_act
#' @template pipeop_torch_format
#'
#' @description
#' Base class for activation functions.
#'
#' @section Module:
#' See the respective child class.
#'
#' @template pipeop_torch_channels_default
#'
#' @template pipeop_torch_state_default
#'
#' @section Parameters:
#' See the respective child class.
#'
#' @template torch_license_docu
#'
#' @family PipeOpTorch
#'
#' @template param_id
#' @template param_param_vals
#' @template param_param_set
#' @template param_module_generator
#'
#' @export
PipeOpTorchActivation = R6Class("PipeOpTorchActivation",
  inherit = PipeOpTorch,
  public = list(
    #' @description Initializes an instance of this [R6][R6::R6Class] class.
    initialize = function(id, param_vals = list(), param_set = ps(), module_generator) {
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals,
        module_generator = module_generator
      )
    }
  )
)

#' @title Activation Elu
#'
#' @usage NULL
#' @name pipeop_torch_act_elu
#' @template pipeop_torch_format
#'
#' @inherit torch::nnf_elu description
#'
#' @section Module:
#' Calls [`torch::nn_elu()`] during training.
#'
#' @template pipeop_torch_channels_default
#'
#' @template pipeop_torch_state_default
#'
#' @section Parameters:
#' * `alpha` :: `numeric(1)`\cr
#'   The alpha value for the ELU formulation. Default: 1.0
#' * `inplace` :: `logical(1)`\cr
#'   Whether to do the operation in-place. Default: `FALSE`.
#'
#' @template torch_license_docu
#' @family PipeOpTorch
#' @template param_id
#' @template param_param_vals
#' @export
PipeOpTorchActivationElu = R6Class("PipeOpTorchActivationElu",
  inherit = PipeOpTorchActivation,
  public = list(
    #' @description Initializes an instance of this [R6][R6::R6Class] class.
    initialize = function(id = "nn_act_elu", param_vals = list()) {
      param_set = ps(
        alpha = p_dbl(default = 1, tags = "train"),
        inplace = p_lgl(default = FALSE, tags = "train")
      )
      super$initialize(id = id, param_vals = param_vals, param_set = param_set, module_generator = nn_elu)
    }
  )
)

#' @title Activation Elu
#'
#' @usage NULL
#' @name pipeop_torch_act_elu
#' @template pipeop_torch_format
#'
#' @inherit torch::nnf_elu description
#'
#' @section Module:
#' Calls [`torch::nn_elu()`] during training.
#'
#' @template pipeop_torch_channels_default
#'
#' @template pipeop_torch_state_default
#'
#' @section Parameters:
#' * `alpha` :: `numeric(1)`\cr
#'   The alpha value for the ELU formulation. Default: 1.0
#' * `inplace` :: `logical(1)`\cr
#'   Whether to do the operation in-place. Default: `FALSE`.
#'
#' @template torch_license_docu
#' @family PipeOpTorch
#' @template param_id
#' @template param_param_vals
#' @export
PipeOpTorchActivationHardShrink = R6Class("PipeOpTorchActivationHardShrink",
  inherit = PipeOpTorchActivation,
  public = list(
    #' @description Initializes an instance of this [R6][R6::R6Class] class.
    initialize = function(id = "nn_act_hardshrink", param_vals = list()) {
      param_set = ps(
        lambd = p_dbl(default = 0.5, tags = "train")
      )
      super$initialize(id = id, param_vals = param_vals, param_set = param_set)
    }
  )
)

#' @title Activation Elu
#'
#' @usage NULL
#' @name pipeop_torch_act_elu
#' @template pipeop_torch_format
#'
#' @inherit torch::nnf_elu description
#'
#' @section Module:
#' Calls [`torch::nn_elu()`] during training.
#'
#' @template pipeop_torch_channels_default
#'
#' @template pipeop_torch_state_default
#'
#' @section Parameters:
#' * `alpha` :: `numeric(1)`\cr
#'   The alpha value for the ELU formulation. Default: 1.0
#' * `inplace` :: `logical(1)`\cr
#'   Whether to do the operation in-place. Default: `FALSE`.
#'
#' @template torch_license_docu
#' @family PipeOpTorch
#' @template param_id
#' @template param_param_vals
#' @export
PipeOpTorchActivationHardSigmoid = R6Class("PipeOpTorchActivationHardSigmoid",
  inherit = PipeOpTorchActivation,
  public = list(
    #' @description Initializes an instance of this [R6][R6::R6Class] class.
    initialize = function(id = "nn_act_hard_sigmoid", param_vals = list()) {
      super$initialize(id = id, param_vals = param_vals, module_generator = nn_hard_sigmoid)
    }
  )
)

#' @title Activation Elu
#'
#' @usage NULL
#' @name pipeop_torch_act_elu
#' @template pipeop_torch_format
#'
#' @inherit torch::nnf_elu description
#'
#' @section Module:
#' Calls [`torch::nn_elu()`] during training.
#'
#' @template pipeop_torch_channels_default
#'
#' @template pipeop_torch_state_default
#'
#' @section Parameters:
#' * `alpha` :: `numeric(1)`\cr
#'   The alpha value for the ELU formulation. Default: 1.0
#' * `inplace` :: `logical(1)`\cr
#'   Whether to do the operation in-place. Default: `FALSE`.
#'
#' @template torch_license_docu
#' @family PipeOpTorch
#' @template param_id
#' @template param_param_vals
#' @export
PipeOpTorchActivationTanh = R6Class("PipeOpTorchActivationTanh",
  inherit = PipeOpTorchActivation,
  public = list(
    #' @description Initializes an instance of this [R6][R6::R6Class] class.
    initialize = function(id = "nn_act_tanh", param_vals = list()) {
      param_set = ps(
        min_val = p_dbl(default = -1, tags = "train"),
        max_val = p_dbl(default = 1, tags = "train"),
        inplace = p_lgl(default = FALSE, tags = "train")
      )
      super$initialize(id = id, param_vals = param_vals, param_set = param_set, module_generator = nn_hardtanh)
    }
  )
)

#' @title Activation Elu
#'
#' @usage NULL
#' @name pipeop_torch_act_elu
#' @template pipeop_torch_format
#'
#' @inherit torch::nnf_elu description
#'
#' @section Module:
#' Calls [`torch::nn_elu()`] during training.
#'
#' @template pipeop_torch_channels_default
#'
#' @template pipeop_torch_state_default
#'
#' @section Parameters:
#' * `alpha` :: `numeric(1)`\cr
#'   The alpha value for the ELU formulation. Default: 1.0
#' * `inplace` :: `logical(1)`\cr
#'   Whether to do the operation in-place. Default: `FALSE`.
#'
#' @template torch_license_docu
#' @family PipeOpTorch
#' @template param_id
#' @template param_param_vals
#' @export
PipeOpTorchActivationHardSwish = R6Class("PipeOpTorchActivationHardSwish",
  inherit = PipeOpTorchActivation,
  public = list(
    #' @description Initializes an instance of this [R6][R6::R6Class] class.
    initialize = function(id = "nn_act_relu", param_vals = list()) {
      super$initialize(id = id, param_vals = param_vals, param_set = param_set, module_generator = nn_hardswish)
    }
  )
)

#' @title Activation Elu
#'
#' @usage NULL
#' @name pipeop_torch_act_elu
#' @template pipeop_torch_format
#'
#' @inherit torch::nnf_elu description
#'
#' @section Module:
#' Calls [`torch::nn_elu()`] during training.
#'
#' @template pipeop_torch_channels_default
#'
#' @template pipeop_torch_state_default
#'
#' @section Parameters:
#' * `alpha` :: `numeric(1)`\cr
#'   The alpha value for the ELU formulation. Default: 1.0
#' * `inplace` :: `logical(1)`\cr
#'   Whether to do the operation in-place. Default: `FALSE`.
#'
#' @template torch_license_docu
#' @family PipeOpTorch
#' @template param_id
#' @template param_param_vals
#' @export
PipeOpTorchActivationRelu = R6Class("PipeOpTorchActivationRelu",
  inherit = PipeOpTorchActivation,
  public = list(
    #' @description Initializes an instance of this [R6][R6::R6Class] class.
    initialize = function(id = "nn_act_relu", param_vals = list()) {
      param_set = ps(
        negative_slope = p_dbl(default = 0.01, tags = "train"),
        inplace = p_lgl(default = FALSE, tags = "train")
      )
      super$initialize(id = id, param_vals = param_vals, param_set = param_set)
    }
  )
)

#' @title Activation Elu
#'
#' @usage NULL
#' @name pipeop_torch_act_elu
#' @template pipeop_torch_format
#'
#' @inherit torch::nnf_elu description
#'
#' @section Module:
#' Calls [`torch::nn_elu()`] during training.
#'
#' @template pipeop_torch_channels_default
#'
#' @template pipeop_torch_state_default
#'
#' @section Parameters:
#' * `alpha` :: `numeric(1)`\cr
#'   The alpha value for the ELU formulation. Default: 1.0
#' * `inplace` :: `logical(1)`\cr
#'   Whether to do the operation in-place. Default: `FALSE`.
#'
#' @template torch_license_docu
#' @family PipeOpTorch
#' @template param_id
#' @template param_param_vals
#' @export
PipeOpTorchActivationLogSigmoid = R6Class("PipeOpTorchActivationLogSigmoid",
  inherit = PipeOpTorchActivation,
  public = list(
    #' @description Initializes an instance of this [R6][R6::R6Class] class.
    initialize = function(id = "nn_act_log_sigmoid", param_vals = list()) {
      super$initialize(id = id, param_vals = param_vals, module_generator = nn_log_sigmoid)
    }
  )
)

#' @title Activation Elu
#'
#' @usage NULL
#' @name pipeop_torch_act_elu
#' @template pipeop_torch_format
#'
#' @inherit torch::nnf_elu description
#'
#' @section Module:
#' Calls [`torch::nn_elu()`] during training.
#'
#' @template pipeop_torch_channels_default
#'
#' @template pipeop_torch_state_default
#'
#' @section Parameters:
#' * `alpha` :: `numeric(1)`\cr
#'   The alpha value for the ELU formulation. Default: 1.0
#' * `inplace` :: `logical(1)`\cr
#'   Whether to do the operation in-place. Default: `FALSE`.
#'
#' @template torch_license_docu
#' @family PipeOpTorch
#' @template param_id
#' @template param_param_vals
#' @export
PipeOpTorchActivationPrelu = R6Class("PipeOpTorchActivationPrelu",
  inherit = PipeOpTorchActivation,
  public = list(
    #' @description Initializes an instance of this [R6][R6::R6Class] class.
    initialize = function(id = "nn_act_prelu", param_vals = list()) {
      param_set = ps(
        num_parameters = p_int(1, tags = "train"),
        init = p_dbl(default = 0.25, tags = "train")
      )
      super$initialize(id = id, param_vals = param_vals, param_set = param_set, module_generator = nn_prelu)
    }
  )
)

#' @title Activation Elu
#'
#' @usage NULL
#' @name pipeop_torch_act_elu
#' @template pipeop_torch_format
#'
#' @inherit torch::nnf_elu description
#'
#' @section Module:
#' Calls [`torch::nn_elu()`] during training.
#'
#' @template pipeop_torch_channels_default
#'
#' @template pipeop_torch_state_default
#'
#' @section Parameters:
#' * `alpha` :: `numeric(1)`\cr
#'   The alpha value for the ELU formulation. Default: 1.0
#' * `inplace` :: `logical(1)`\cr
#'   Whether to do the operation in-place. Default: `FALSE`.
#'
#' @template torch_license_docu
#' @family PipeOpTorch
#' @template param_id
#' @template param_param_vals
#' @export
PipeOpTorchActivationRelu = R6Class("PipeOpTorchActivationRelu",
  inherit = PipeOpTorchActivation,
  public = list(
    #' @description Initializes an instance of this [R6][R6::R6Class] class.
    initialize = function(id = "nn_act_relu", param_vals = list()) {
      param_set = ps( # nolint
        inplace = p_lgl(default = FALSE, tags = "train")
      )
      super$initialize(id = id, param_vals = param_vals, param_set = param_set, module_generator = nn_relu)
    }
  )
)

#' @title Activation Elu
#'
#' @usage NULL
#' @name pipeop_torch_act_elu
#' @template pipeop_torch_format
#'
#' @inherit torch::nnf_elu description
#'
#' @section Module:
#' Calls [`torch::nn_elu()`] during training.
#'
#' @template pipeop_torch_channels_default
#'
#' @template pipeop_torch_state_default
#'
#' @section Parameters:
#' * `alpha` :: `numeric(1)`\cr
#'   The alpha value for the ELU formulation. Default: 1.0
#' * `inplace` :: `logical(1)`\cr
#'   Whether to do the operation in-place. Default: `FALSE`.
#'
#' @template torch_license_docu
#' @family PipeOpTorch
#' @template param_id
#' @template param_param_vals
#' @export
PipeOpTorchActivationRelu6 = R6Class("PipeOpTorchActivationRelu6",
  inherit = PipeOpTorchActivation,
  public = list(
    #' @description Initializes an instance of this [R6][R6::R6Class] class.
    initialize = function(id = "nn_act_relu6", param_vals = list()) {
      param_set = ps( # nolint
        inplace = p_lgl(default = FALSE, tags = "train")
      )
      super$initialize(id = id, param_vals = param_vals, param_set = param_set, module_generator = nn_relu6)
    }
  )
)

#' @title Activation Rrelu
#'
#' @usage NULL
#' @name pipeop_torch_act_rrelu
#' @template pipeop_torch_format
#'
#' @inherit torch::nnf_rrelu description
#'
#' @section Module:
#' Calls [`torch::nn_rrelu()`] during training.
#'
#' @template pipeop_torch_channels_default
#'
#' @template pipeop_torch_state_default
#'
#' @section Parameters:
#' * `lower`:: `numeric(1)`\cr
#'   Lower bound of the uniform distribution. Default: 1/8.
#' * `upper`:: `numeric(1)`\cr
#'   Upper bound of the uniform distribution. Default: 1/3.
#' * `inplace` :: `logical(1)`\cr
#'   Whether to do the operation in-place. Default: `FALSE`.
#'
#' @template torch_license_docu
#' @family PipeOpTorch
#' @template param_id
#' @template param_param_vals
#' @export
PipeOpTorchActivationRrelu = R6Class("PipeOpTorchActivationRrelu",
  inherit = PipeOpTorchActivation,
  public = list(
    #' @description Initializes an instance of this [R6][R6::R6Class] class.
    initialize = function(id = "nn_act_rrelu", param_vals = list()) {
      param_set = ps(
        lower = p_dbl(default = 1 / 8, tags = "train"),
        upper = p_dbl(default = 1 / 3, tags = "train"),
        inplace = p_lgl(default = FALSE, tags = "train")
      )
      super$initialize(id = id, param_vals = param_vals, param_set = param_set, module_generator = nn_rrelu)
    }
  )
)

#' @title Activation Selu
#'
#' @usage NULL
#' @name pipeop_torch_act_selu
#' @template pipeop_torch_format
#'
#' @inherit torch::nnf_selu description
#'
#' @section Module:
#' Calls [`torch::nn_selu()`] during training.
#'
#' @template pipeop_torch_channels_default
#'
#' @template pipeop_torch_state_default
#'
#' @section Parameters:
#' * `inplace` :: `logical(1)`\cr
#'   Whether to do the operation in-place. Default: `FALSE`.
#'
#' @template torch_license_docu
#' @family PipeOpTorch
#' @template param_id
#' @template param_param_vals
#' @export
PipeOpTorchActivationSelu = R6Class("PipeOpTorchActivationSelu",
  inherit = PipeOpTorchActivation,
  public = list(
    #' @description Initializes an instance of this [R6][R6::R6Class] class.
    initialize = function(id = "nn_act_selu", param_vals = list()) {
      param_set = ps(
        inplace = p_lgl(default = FALSE, tags = "train")
      )
      super$initialize(id = id, param_vals = param_vals, param_set = param_set, module_generator = nn_selu)
    }
  )
)

#' @title Activation Celu
#'
#' @usage NULL
#' @name pipeop_torch_act_celu
#' @template pipeop_torch_format
#'
#' @inherit torch::nnf_celu description
#'
#' @section Module:
#' Calls [`torch::nn_celu()`] during training.
#'
#' @template pipeop_torch_channels_default
#'
#' @template pipeop_torch_state_default
#'
#' @section Parameters:
#' * `alpha` :: `numeric(1)`\cr
#'   The alpha value for the ELU formulation. Default: 1.0
#' * `inplace` :: `logical(1)`\cr
#'   Whether to do the operation in-place. Default: `FALSE`.
#'
#' @template torch_license_docu
#' @family PipeOpTorch
#' @template param_id
#' @template param_param_vals
#' @export
PipeOpTorchActivationCelu = R6Class("PipeOpTorchActivationCelu",
  inherit = PipeOpTorchActivation,
  public = list(
    #' @description Initializes an instance of this [R6][R6::R6Class] class.
    initialize = function(id = "nn_act_", param_vals = list()) {
      param_set = ps( # nolint
        alpha = p_dbl(default = 1.0, tags = "train"),
        inplace = p_lgl(default = FALSE, tags = "train")
      )
      super$initialize(id = id, param_vals = param_vals, param_set = param_set, module_generator = nn_celu)
    }
  )
)

#' @title Activation Felu
#'
#' @usage NULL
#' @name pipeop_torch_act_gelu
#' @template pipeop_torch_format
#'
#' @inherit torch::nnf_gelu description
#'
#' @section Module:
#' Calls [`torch::nn_gelu()`] during training.
#'
#' @template pipeop_torch_channels_default
#'
#' @template pipeop_torch_state_default
#'
#' @section Parameters:
#' No parameters.
#'
#' @template torch_license_docu
#' @family PipeOpTorch
#' @template param_id
#' @template param_param_vals
#' @export
PipeOpTorchActivationGelu = R6Class("PipeOpTorchActivationGelu",
  inherit = PipeOpTorchActivation,
  public = list(
    #' @description Initializes an instance of this [R6][R6::R6Class] class.
    initialize = function(id = "nn_act_gelu", param_vals = list()) {
      super$initialize(id = id, param_vals = param_vals, module_generator = nn_gelu)
    }
  )
)

#' @title Activation Sigmoid
#'
#' @usage NULL
#' @name pipeop_torch_act_sigmoid
#' @template pipeop_torch_format
#'
#' @inherit torch::nnf_sigmoid description
#'
#' @section Module:
#' Calls [`torch::nn_sigmoid()`] during training.
#'
#' @template pipeop_torch_channels_default
#'
#' @template pipeop_torch_state_default
#'
#' @section Parameters:
#' No parameters.
#'
#' @template torch_license_docu
#' @family PipeOpTorch
#' @template param_id
#' @template param_param_vals
#' @export
PipeOpTorchActivationSigmoid = R6Class("PipeOpTorchActivationSigmoid",
  inherit = PipeOpTorchActivation,
  public = list(
    #' @description Initializes an instance of this [R6][R6::R6Class] class.
    initialize = function(id = "nn_act_sigmoid", param_vals = list()) {
      super$initialize(id = id, param_vals = param_val, module_generator = nn_sigmoid)
    }
  )
)

#' @title Activation SoftPlus
#'
#' @usage NULL
#' @name pipeop_torch_act_softplus
#' @template pipeop_torch_format
#'
#' @inherit torch::nnf_softplus description
#'
#' @section Module:
#' Calls [`torch::nn_softplus()`] during training.
#'
#' @template pipeop_torch_channels_default
#'
#' @template pipeop_torch_state_default
#'
#' @section Parameters:
#' * `beta` :: `numeric(1)`\cr
#'   The beta value for the Softplus formulation. Default: 1
#' * `threshold` :: `numeric(1)`\cr
#'   Values above this revert to a linear function. Default: 20
#'
#' @template torch_license_docu
#' @family PipeOpTorch
#' @template param_id
#' @template param_param_vals
#' @export
PipeOpTorchActivationSoftPlus = R6Class("PipeOpTorchActivationSoftPlus",
  inherit = PipeOpTorchActivation,
  public = list(
    #' @description Initializes an instance of this [R6][R6::R6Class] class.
    initialize = function(id = "nn_act_softplus", param_vals = list()) {
      param_set = ps(
        beta = p_dbl(default = 1, tags = "train"),
        threshold = p_dbl(default = 20, tags = "train")
      )
      super$initialize(id = id, param_vals = param_vals, param_set = param_set, module_generator = nn_softplus)
    }
  )
)

#' @title Activation SoftShrink
#'
#' @usage NULL
#' @name pipeop_torch_act_softshrink
#' @template pipeop_torch_format
#'
#' @inherit torch::nnf_softshrink description
#'
#' @section Module:
#' Calls [`torch::nn_softshrink()`] during training.
#'
#' @template pipeop_torch_channels_default
#'
#' @template pipeop_torch_state_default
#'
#' @section Parameters:
#' * `lamd` :: `numeric(1)`\cr
#'   The lambda (must be no less than zero) value for the Softshrink formulation. Default: 0.5
#'
#' @template torch_license_docu
#' @family PipeOpTorch
#' @template param_id
#' @template param_param_vals
#' @export
PipeOpTorchActivationSoftShrink = R6Class("PipeOpTorchActivationSoftShrink",
  inherit = PipeOpTorchActivation,
  public = list(
    #' @description Initializes an instance of this [R6][R6::R6Class] class.
    initialize = function(id = "nn_act_softshrink", param_vals = list()) {
      param_set = ps(
        lambd = p_dbl(default = 0.5, upper = 1, tags = "train")
      )
      super$initialize(id = id, param_vals = param_vals, param_set = param_set, module_generator = nn_softshtink)
    }
  )
)

#' @title Activation SoftSign
#'
#' @usage NULL
#' @name pipeop_torch_act_elu
#' @template pipeop_torch_format
#'
#' @inherit torch::nnf_softsign description
#'
#' @section Module:
#' Calls [`torch::nn_softsign()`] during training.
#'
#' @template pipeop_torch_channels_default
#'
#' @template pipeop_torch_state_default
#'
#' @section Parameters:
#' No parameters.
#'
#' @template torch_license_docu
#' @family PipeOpTorch
#' @template param_id
#' @template param_param_vals
#' @export
PipeOpTorchActivationSoftSign = R6Class("PipeOpTorchActivationSoftSign",
  inherit = PipeOpTorchActivation,
  public = list(
    #' @description Initializes an instance of this [R6][R6::R6Class] class.
    initialize = function(id = "nn_act_softsign", param_vals = list()) {
      super$initialize(id = id, param_vals = param_vals, module_generator = nn_softsign)
    }
  )
)

#' @title Activation Tanh
#'
#' @usage NULL
#' @name pipeop_torch_act_tanh
#' @template pipeop_torch_format
#'
#' @inherit torch::torch_tanh description
#'
#' @section Module:
#' Calls [`torch::nn_tanh()`] during training.
#'
#' @template pipeop_torch_channels_default
#'
#' @template pipeop_torch_state_default
#'
#' @section Parameters:
#' No parameters.
#'
#' @template torch_license_docu
#' @family PipeOpTorch
#' @template param_id
#' @template param_param_vals
#' @export
PipeOpTorchActivationTanh = R6Class("PipeOpTorchActivationTanh",
  inherit = PipeOpTorchActivation,
  public = list(
    #' @description Initializes an instance of this [R6][R6::R6Class] class.
    initialize = function(id = "nn_act_tanh", param_vals = list()) {
      super$initialize(id = id, param_vals = param_vals, module_generator = nn_tanh)
    }
  )
)

#' @title Activation TanhShrink
#'
#' @usage NULL
#' @name pipeop_torch_act_tanhshrink
#' @template pipeop_torch_format
#'
#' @inherit torch::nnf_tanhshrink description
#'
#' @section Module:
#' Calls [`torch::nn_tanhshrink()`] during training.
#'
#' @template pipeop_torch_channels_default
#'
#' @template pipeop_torch_state_default
#'
#' @section Parameters:
#' No parameters.
#'
#' @template torch_license_docu
#' @family PipeOpTorch
#' @template param_id
#' @template param_param_vals
#' @export
PipeOpTorchActivationTanhShrink = R6Class("PipeOpTorchActivationTanhShrink", # nolint
  inherit = PipeOpTorchActivation,
  public = list(
    #' @description Initializes an instance of this [R6][R6::R6Class] class.
    initialize = function(id = "nn_act_tanhshrink", param_vals = list()) {
      super$initialize(id = id, param_vals = param_vals, module_generator = nn_tanhshrink)
    }
  )
)

#' @title Activation GLU
#'
#' @usage NULL
#' @name pipeop_torch_act_glu
#' @template pipeop_torch_format
#'
#' @inherit torch::nnf_glu description
#'
#' @section Module:
#' Calls [`torch::nn_glu()`] during training.
#'
#' @template pipeop_torch_channels_default
#' @template pipeop_torch_state_default
#'
#' @section Parameters:
#' * `dim` :: `integer(1)`\cr
#'   Dimension on which to split the input. Default: -1
#'
#' @template torch_license_docu
#' @family PipeOpTorch
#' @template param_id
#' @template param_param_vals
#' @export
PipeOpTorchActivationGlu = R6Class("PipeOpTorchActivationGlu",
  inherit = PipeOpTorchActivation,
  public = list(
    #' @description Initializes an instance of this [R6][R6::R6Class] class.
    initialize = function(id = "nn_act_glu", param_vals = list()) {
      param_set = ps(
        dim = p_int(default = -1L, lower = 1L, tags = "train", special_vals = list(-1L))
      )
      super$initialize(id = id, param_vals = param_vals, param_set = param_set, module_generator = nn_glu)
    }
  ),
  private = list(
    .shapes_out = function(shapes_in, param_vals) {
      assert_true(length(shapes_in[[1L]]) >= param_vals$dim)
      assert_true(shapes_in[[1]][param_vals$dim] %% 2 == 0)
      shapes_in[[1L]][param_vals$dim] = shapes_in[[1L]][param_vals$dim] / 2
      shapes_in
    }
  )
)
