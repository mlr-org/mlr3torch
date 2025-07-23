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
    #' @param torch_loss (`nn_loss` or `function`)\cr
    #'   The loss module or function that generates the loss module.
    #'   Can have arguments `task` that will be provided when the loss is instantiated.
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
      assert(check_class(torch_loss, "nn_module_generator"), check_class(torch_loss, "function"))

      param_set = assert_r6(param_set, "ParamSet", null.ok = TRUE) %??% inferps(torch_loss, ignore = "task")
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
    },
    #' @description
    #' Instantiates the loss function.
    #' @param task (`Task`)\cr
    #'   The task. Must be provided if the loss function requires a task.
    #' @return `torch_loss`
    generate = function(task = NULL) {
      require_namespaces(self$packages)
      args = self$param_set$get_values()
      if ("task" %in% formalArgs(self$generator)) {
        assert_true(!is.null(task), .var.name = "task is provided if required by loss function")
        args = insert_named(args, list(task = task))
      }
      do.call(self$generator, args)
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
      packages = list(loss$packages)
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
    class_weight = p_uty(default = NULL, tags = "train"),
    ignore_index = p_int(default = -100, tags = "train"),
    reduction = p_fct(levels = c("mean", "sum"), default = "mean", tags = "train")
  )
  TorchLoss$new(
    torch_loss = function(task, ...) {
      if (task$task_type != "classif") {
        stopf("Cross entropy loss is only defined for classification tasks, but task is of type '%s'", task$task_type)
      }
      args = list(...)
      is_binary = "twoclass" %in% task$properties
      if (is_binary) {
        if (!is.null(args$ignore_index)) {
          stopf("ignore_index is not supported for binary cross entropy loss")
        }
        if (!is.null(args$class_weight)) {
          args$pos_weight = args$class_weight
          args$class_weight = NULL
        }
        return(invoke(nn_bce_with_logits_loss, .args = args))
      }
      if (!is.null(args$class_weight)) {
        args$weight = args$class_weight
        args$class_weight = NULL
      }
      invoke(nn_cross_entropy_loss, .args = args)
    },
    task_types = "classif",
    param_set = p,
    id = "cross_entropy",
    label = "Cross Entropy",
    man = "mlr3torch::cross_entropy"
  )
})

#' @title Cross Entropy Loss
#' @name cross_entropy
#' @description
#' The `cross_entropy` loss function selects the multi-class ([`nn_cross_entropy_loss`][torch::nn_cross_entropy_loss])
#' or binary ([`nn_bce_with_logits_loss`][torch::nn_bce_with_logits_loss]) cross entropy
#' loss based on the number of classes.
#' Because of this, there is a slight reparameterization of the loss arguments, see *Parameters*.
#' @section Parameters:
#' * `class_weight`:: [`torch_tensor`][torch::torch_tensor]\cr
#'    The class weights. For multi-class problems, this must be a `torch_tensor` of length `num_classes`
#'    (and is passed as argument `weight` to [`nn_cross_entropy_loss`][torch::nn_cross_entropy_loss]).
#'    For binary problems, this must be a scalar (and is passed as argument `pos_weight` to
#'    [`nn_bce_with_logits_loss`][torch::nn_bce_with_logits_loss]).
#' - `ignore_index`:: `integer(1)`\cr
#'    Index of the class which to ignore and which does not contribute to the gradient.
#'    This is only available for multi-class loss.
#' - `reduction` :: `character(1)`\cr
#'    The reduction to apply. Is either `"mean"` or `"sum"` and passed as argument `reduction`
#'    to either loss function. The default is `"mean"`.
#' @examplesIf torch::torch_is_installed()
#' loss = t_loss("cross_entropy")
#' # multi-class
#' multi_ce = loss$generate(tsk("iris"))
#' multi_ce
#'
#' # binary
#' binary_ce = loss$generate(tsk("sonar"))
#' binary_ce
NULL
