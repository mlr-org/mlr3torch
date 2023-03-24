#' @title Convert to TorchLoss
#'
#' @description
#' Converts an object to a [`TorchLoss`].
#'
#' @param x (any)\cr
#'   Object to convert to a [`TorchLoss`].
#' @param clone (`logical(1)`\cr
#'   Whether to make a deep clone.
#' @param ... (any)\cr
#' Additional arguments.
#'
#' @return [`TorchLoss`].
#' @export
as_torch_loss = function(x, clone = FALSE, ...) {
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
as_torch_loss.character = function(x, clone = FALSE, ...) { # nolint
  t_loss(x, ...)
}

#' @title Torch Loss
#'
#' @usage NULL
#' @name torch_loss
#' @format [`R6Class`]
#'
#' @description
#' This wraps a `torch::nn_loss` and is usually used to configure
#' It is commonly used to configure the `loss` of a torch learner.
#'
#' For a list of available parameters, seen [`mlr3torch_losses`].
#'
#' @section Construction:
#' `r roxy_construction(TorchLoss)`
#'
#' * `torch_loss` :: `nn_loss`\cr
#'   The loss module.
#' * `param_set` :: (`ParamSet`)\cr
#'   The parameter set of the oss If this is `NULL` (default), the parameter set is inferred, leading to potentially
#'   less precise parameter descriptions.
#' * `task_types` :: `character()`\cr
#'   The task types supported by this loss.
#'   lf left `NULL` (default), this value is set to all available task types.
#' * `label` :: `character(1)`\cr
#'   The label for the `TorchLoss`.
#' * `packages` :: `character()`\cr
#'   The packages this loss depends on. The values `"torch"` and `"mlr3torch"` are always included. 
#'   Default is `NULL`.
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
#' * `help()`\cr
#'   Opens the help page for the wrapped loss.
#'
#' @family torch_wrapper
#' @export
#' @examples
#' # Create a new Torch Loss
#' torchloss = TorchLoss$new(torch_loss = nn_mse_loss, task_types = "regr")
#' # If the param set is not specified, parameters are inferred but are of class ParamUty
#'
#' # Open help page
#' # torchloss$help()
#'
#' loss$param_set
#' # Construct the actual loss function
#' los = tchloss$get_loss()
TorchLoss = R6::R6Class("TorchLoss",
  public = list(
    label = NULL,
    task_types = NULL,
    loss = NULL,
    param_set = NULL,
    packages = NULL,
    initialize = function(torch_loss, task_types = NULL, param_set = NULL, label = deparse(substitute(torch_loss))[[1]],
      packages = NULL) {
      assert_r6(param_set, "ParamSet", null.ok = TRUE)
      task_types = task_types %??% mlr_reflections$task_types$type
      self$task_types = assert_subset(task_types, mlr_reflections$task_types$type)
      self$label = assert_string(label)
      self$loss = assert_class(torch_loss, "nn_loss") # maybe too strict?
      packages = union(packages, c("torch", "mlr3torch"))
      self$packages = assert_names(packages, type = "strict")

      self$param_set = param_set %??% inferps(torch_loss)
    },
    get_loss = function() {
      require_namespaces(self$packages)
      invoke(self$loss, .args = self$param_set$get_values())
    }, 
    print = function(...) {
      catn(sprintf("<TorchLoss:%s>", self$label))
      catn(str_indent("* Generator:", self$optimizer$classname))
      catn(str_indent("* Parameters:", as_short_string(self$param_set$values, 1000L)))
      catn(str_indent("* Task Types:", as_short_string(self$task_types, 1000L)))
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
})


mlr3torch_losses$add("l1", function() {
  p = ps(reduction = p_fct(levels = c("mean", "sum"), default = "mean", tags = "train"))
  TorchLoss$new(torch::nn_l1_loss, "regr", p, "l1")
})


mlr3torch_losses$add("cross_entropy", function() {
  p = ps(
    weight = p_uty(default = NULL, tags = "train"),
    ignore_index = p_int(default = -100L, tags = "train"),
    reduction = p_fct(levels = c("mean", "sum"), default = "mean", tags = "train")
  )
  TorchLoss$new(torch::nn_cross_entropy_loss, "classif", p, "cross_entropy")
})
