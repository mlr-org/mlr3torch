#' @title Convert to TorchLoss
#' @description
#' Converts an object to a [`TorchLoss`].
#'
#' @param x (any)\cr
#'   Object to convert to a [`TorchLoss`].
#' @param clone (`logical(1)`\cr
#'   Whether to make a deep clone.
#' @export
as_torch_loss = function(x, clone = FALSE) {
  assert_flag(clone)
  UseMethod("as_torch_loss")
}

#' @export
as_torch_loss.nn_loss = function(x, clone = FALSE) { # nolint
  TorchLoss$new(x, label = deparse(substitute(x))[[1]])
}

#' @export
as_torch_loss.TorchLoss = function(x, clone = FALSE) { # nolint
  if (clone) x$clone(deep = TRUE) else x
}

#' @export
as_torch_loss.character = function(x, clone = FALSE) { # nolint
  t_loss(x)
}

as_torch_loss.default = function(x, clone) { # nolint 
  stopf("Cannot convert object of type %s to loss.", class(x)[[1L]])
}

#' @title Torch Loss
#'
#' @usage NULL
#' @name torch_loss
#' @format [`R6Class`]
#'
#' @description
#' This wraps a `torch::nn_loss`.
#' Can be used to configure the `loss` of a [`ModelDescriptor`]..
#' For a list of available parameters, seen [`mlr3torch_losses`].
#'
#' @section Construction:
#' `r roxy_construction(PipeOpTorchLoss)`
#'
#' * `r roxy_param_param_set()`
#' * `torch_loss` :: `nn_loss`\cr
#'   The loss module.
#' * `task_types` :: `character()`\cr
#'   The task types supported by this loss.
#' * `label` :: `character(1)`\cr
#'   The label for the `TorchLoss`.
#' * `packages` :: `character()`\cr
#'   The packages this loss depends on.
#'
#' @section Parameters:
#' Defined by the constructor argument `param_set`.
#'
#' @section Fields:
#' * `label` :: `character(1)`\cr
#'  The label for the object.
#' * `task_types` :: `character()`\cr
#'  The task types that are supported.
#' * `loss` :: `
#'   The generator of the loss function.
#' * `param_set` :: `paradox::ParamSet`\cr
#'   The parameter set.
#' * `packages` :: `character()`\cr
#'   The packages this loss requires.
#'
#' @section Methods:
#' * `get_loss()`\cr
#'   () -> `nn_loss()`\cr
#'   Initializes the torch loss for the given parameter values.
#'
#' @family TorchOptimizer
#' @export
#' @examples
#' # Create a new Torch Loss
#' loss = TorchLoss$new(torch_loss = torch::nn_mse_loss, task_types = "regr")
#' # If the param set is not specified, parameters are inferred but are of class ParamUty
#' loss$param_set
#' # Construct the actual loss function
#' l = loss$get_loss
TorchLoss = R6::R6Class("TorchLoss",
  public = list(
    label = NULL,
    task_types = NULL,
    loss = NULL,
    param_set = NULL,
    packages = NULL,
    initialize = function(torch_loss, task_types, param_set = NULL, label = deparse(substitute(torch_loss))[[1]],
      packages = "torch") {
      assert_r6(param_set, "ParamSet", null.ok = TRUE)
      self$task_types = assert_subset(task_types, mlr_reflections$task_types$type)
      self$label = assert_string(label)
      self$loss = assert_class(torch_loss, "nn_loss")  # maybe too strict?
      self$packages = assert_names(packages, type = "strict")

      self$param_set = param_set %??% inferps(torch_loss)
    },
    get_loss = function() {
      invoke(self$loss, .args = self$param_set$get_values())
    }
  )
)

#' @title Loss Functions
#' @description
#' Dictionary of torch loss functions
#' See [`t_loss`] for conveniently retrieving a loss function.
#'
#' @section Available Loss Functions:
#'
#' * mse - [`torch::nn_mse_loss`]
#' * l1 - [`torch::nn_l1_loss`]
#' * cross_entropy - [`torch::nn_cross_entropy_loss`]
#' @export
mlr3torch_losses = R6Class("DictionaryMlr3torchLosses",
  inherit = Dictionary,
  cloneable = FALSE
)$new()


#' @title Loss Function Quick Access
#' @param .key (`character(1)`)\cr
#'   Key of the object to retrieve.
#' @param ... (any)\cr
#'   See description of [`dictionary_sugar_get`].
#' @return A [`TorchLoss`]
#' @examples
#' torch_loss = t_loss("mse")
#' torch_loss$param_set
#' torch_loss$loss
#' @export
t_loss = function(.key, ...) {
  dictionary_sugar_get(mlr3torch_losses, .key, ...)
}

mlr3torch_losses$add("mse", function() {
    p = ps(reduction = p_fct(levels = c("mean", "sum"), default = "mean", tags = "train"))
    TorchLoss$new(torch::nn_mse_loss, "regr", p, "mse")
  }
)


mlr3torch_losses$add("l1", function() {
    p = ps()
    TorchLoss$new(torch::nn_l1_loss, "regr", p, "l1")
  }
)


mlr3torch_losses$add("cross_entropy", function() {
    p = ps(
      weight = p_uty(),
      ignore_index = p_int(),
      reduction = p_fct(levels = c("mean", "sum"), default = "mean")
    )
    TorchLoss$new(torch::nn_cross_entropy_loss, "classif", p, "cross_entropy")
  }
)
