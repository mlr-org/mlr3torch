#' @title Convert to TorchLoss
#'
#' @description
#' Converts an object to a [`TorchLoss`].
#'
#' @param x (any)\cr
#'   Object to convert to a [`TorchLoss`].
#' @param clone (`logical(1)`)\cr
#'   Whether to make a deep clone.
#' @param ... (any)\cr
#'   Additional arguments.
#'   Currently used to pass additional constructor arguments to [`TorchLoss`] for objects of type `nn_loss`.
#'
#' @family Torch Descriptor
#'
#' @return [`TorchLoss`].
#' @export
as_torch_loss = function(x, clone = FALSE, ...) {
  assert_flag(clone)
  UseMethod("as_torch_loss")
}

#' @export
as_torch_loss.nn_loss = function(x, clone = FALSE, ...) { # nolint
  # clone argument is irrelevant
  TorchLoss$new(x, ...)
}

#' @export
as_torch_loss.TorchLoss = function(x, clone = FALSE, ...) { # nolint
  if (clone) x$clone(deep = TRUE) else x
}

#' @export
as_torch_loss.character = function(x, clone = FALSE, ...) { # nolint
  t_loss(x, ...)
}

#' @title Torch Loss
#'
#' @description
#' This wraps a `torch::nn_loss` and annotates it with metadata, most importantly a [`ParamSet`][paradox::ParamSet].
#' The loss function is created for the given parameter values by calling the `$generate()` method.
#'
#' This class is usually used to configure the loss function of a torch learner, e.g.
#' when construcing a learner or in a [`ModelDescriptor`].
#'
#' For a list of available losses, see [`mlr3torch_losses`].
#' Items from this dictionary can be retrieved using [`t_loss()`].
#'
#' @section Parameters:
#' Defined by the constructor argument `param_set`.
#' If no parameter set is provided during construction, the parameter set is constructed by creating a parameter
#' for each argument of the wrapped loss function, where the parametes are then of type `ParamUty`.
#'
#' @family Torch Descriptor
#' @export
#' @examplesIf torch::torch_is_installed()
#' # Create a new torch loss
#' torch_loss = TorchLoss$new(torch_loss = nn_mse_loss, task_types = "regr")
#' torch_loss
#' # the parameters are inferred
#' torch_loss$param_set
#'
#' # Retrieve a loss from the dictionary:
#' torch_loss = t_loss("mse", reduction = "mean")
#' # is the same as
#' torch_loss
#' torch_loss$param_set
#' torch_loss$label
#' torch_loss$task_types
#' torch_loss$id
#'
#' # Create the loss function
#' loss_fn = torch_loss$generate()
#' loss_fn
#' # Is the same as
#' nn_mse_loss(reduction = "mean")
#'
#' # open the help page of the wrapped loss function
#' # torch_loss$help()
#'
#' # Use in a learner
#' learner = lrn("regr.mlp", loss = t_loss("mse"))
#' # The parameters of the loss are added to the learner's parameter set
#' learner$param_set
TorchLoss = R6::R6Class("TorchLoss",
  inherit = TorchDescriptor,
  public = list(
    #' @field task_types (`character()`)\cr
    #'  The task types this loss supports.
    task_types = NULL,
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    #' @param torch_loss (`nn_loss`)\cr
    #'   The loss module.
    #' @param task_types (`character()`)\cr
    #'   The task types supported by this loss.
    #' @param param_set ([`ParamSet`][paradox::ParamSet] or `NULL`)\cr
    #'   The parameter set. If `NULL` (default) it is inferred from `torch_loss`.
    #' @template param_id
    #' @template param_label
    #' @template param_packages
    #' @template param_man
    initialize = function(torch_loss, task_types = NULL, param_set = NULL,
      id = NULL, label = NULL, packages = NULL, man = NULL) {
      force(id)
      self$task_types = if (!is.null(task_types)) {
        assert_subset(task_types, mlr_reflections$task_types$type)
      } else {
        c("classif", "regr")
      }
      torch_loss = assert_class(torch_loss, "nn_module")

      super$initialize(
        generator = torch_loss,
        id = id,
        param_set = param_set,
        packages = packages,
        label = label,
        man = man
      )
    },
    #' @description
    #' Prints the object
    #' @param ... any
    print = function(...) {
      super$print(...)
      catn(str_indent("* Task Types:", as_short_string(self$task_types, 1000L)))
      invisible(self)
    }
  ),
  private = list(
    .additional_phash_input = function() {
      self$task_types
    }
  )
)

#' @title Loss Functions
#'
#' @description
#' Dictionary of torch loss descriptors.
#' See [`t_loss()`] for conveniently retrieving a loss function.
#' Can be converted to a [`data.table`][data.table::data.table] using
#' [`as.data.table`][data.table::as.data.table].
#'
#' @section Available Loss Functions:
#' `r paste0(mlr3torch_losses$keys(), collapse = ", ")`
#'
#' @family Torch Descriptor
#' @family Dictionary
#' @export
#' @examplesIf torch::torch_is_installed()
#' mlr3torch_losses$get("mse")
#' # is equivalent to
#' t_loss("mse")
#' # convert to a data.table
#' as.data.table(mlr3torch_losses)
mlr3torch_losses = R6Class("DictionaryMlr3torchLosses",
  inherit = Dictionary,
  cloneable = FALSE
)$new()

#' @export
as.data.table.DictionaryMlr3torchLosses = function(x, ...) {
  setkeyv(map_dtr(x$keys(), function(key) {
    loss = x$get(key)
    list(
      key = key,
      label = loss$label,
      task_types = list(loss$task_types),
      packages = paste0(loss$packages, collapse = ",")
    )}), "key")[]
}



#' @title Loss Function Quick Access
#'
#' @description
#' Retrieve one or more [`TorchLoss`]es from [`mlr3torch_losses`].
#' Works like [`mlr3::lrn()`] and [`mlr3::lrns()`].
#'
#' @param .key (`character(1)`)\cr
#'   Key of the object to retrieve.
#' @param ... (any)\cr
#'   See description of [`dictionary_sugar_get`][mlr3misc::dictionary_sugar_get].
#' @return A [`TorchLoss`]
#' @export
#' @family Torch Descriptor
#' @examplesIf torch::torch_is_installed()
#' t_loss("mse", reduction = "mean")
#' # get the dictionary
#' t_loss()
t_loss = function(.key, ...) {
  UseMethod("t_loss")
}

#' @export
t_loss.character = function(.key, ...) { # nolint
  dictionary_sugar_inc_get(dict = mlr3torch_losses, .key, ...)
}

#' @export
t_loss.NULL = function(.key, ...) { # nolint
  # class is NULL if .key is missing
  dictionary_sugar_get(mlr3torch_losses)
}

#' @rdname t_loss
#' @param .keys (`character()`)\cr
#'   The keys of the losses.
#' @export
#' @examplesIf torch::torch_is_installed()
#' t_losses(c("mse", "l1"))
#' # get the dictionary
#' t_losses()
t_losses = function(.keys, ...) {
  UseMethod("t_losses")
}

#' @export
t_losses.character = function(.keys, ...) { # nolint
  dictionary_sugar_inc_mget(dict = mlr3torch_losses, .keys, ...)
}

#' @export
t_losses.NULL = function(.keys, ...) { # nolint
  # class is NULL if .keys is missing
  dictionary_sugar_get(mlr3torch_losses)

}

mlr3torch_losses$add("mse", function() {
  p = ps(reduction = p_fct(levels = c("mean", "sum"), default = "mean", tags = "train"))
  TorchLoss$new(
    torch_loss = torch::nn_mse_loss,
    task_types = "regr",
    param_set = p,
    id = "mse",
    label = "Mean Squared Error",
    man = "torch::nn_mse_loss"
  )
})


mlr3torch_losses$add("l1", function() {
  p = ps(reduction = p_fct(levels = c("mean", "sum"), default = "mean", tags = "train"))
  TorchLoss$new(
    torch_loss = torch::nn_l1_loss,
    task_types = "regr",
    param_set = p,
    id = "l1",
    label = "Absolute Error",
    man = "torch::nn_l1_loss"
  )
})

mlr3torch_losses$add("cross_entropy", function() {
  p = ps(
    weight = p_uty(default = NULL, tags = "train"),
    ignore_index = p_int(default = -100, tags = "train"),
    reduction = p_fct(levels = c("mean", "sum"), default = "mean", tags = "train")
  )
  TorchLoss$new(
    torch_loss = torch::nn_cross_entropy_loss,
    task_types = "classif",
    param_set = p,
    id = "cross_entropy",
    label = "Cross Entropy",
    man = "torch::nn_cross_entropy_loss"
  )
})
