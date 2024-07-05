#' @title Sugar Function for Torch Callback
#'
#' @description
#' Retrieves one or more [`TorchCallback`]s from [`mlr3torch_callbacks`].
#' Works like [`mlr3::lrn()`] and [`mlr3::lrns()`].
#'
#' @param .key (`character(1)`)\cr
#'   The key of the torch callback.
#' @param ... (any)\cr
#'   See description of [`dictionary_sugar_get()`][mlr3misc::dictionary_sugar_get].
#'
#' @return [`TorchCallback`]
#'
#' @export
#' @family Callback
#' @family Torch Descriptor
#' @examplesIf torch::torch_is_installed()
#' t_clbk("progress")
t_clbk = function(.key, ...) {
  UseMethod("t_clbk")
}

#' @export
t_clbk.character = function(.key, ...) { # nolint
  dictionary_sugar_inc_get(dict = mlr3torch_callbacks, .key = .key, ...)
}

#' @export
t_clbk.NULL = function(.key, ...) { # nolint
  # class is NULL if .key is missing
  dictionary_sugar_get(dict = mlr3torch_callbacks)
}


#' @rdname t_clbk
#' @param .keys (`character()`)\cr
#'   The keys of the callbacks.
#'
#' @return `list()` of [`TorchCallback`]s
#' @export
t_clbks = function(.keys) { # nolint
  UseMethod("t_clbks")
}

#' @export
t_clbks.character = function(.keys, ...) { # nolint
  dictionary_sugar_inc_mget(dict = mlr3torch_callbacks, .keys = .keys, ...)
}

#' @export
t_clbks.NULL = function(.keys, ...) { # nolint
  # class is NULL if .keys is missing
  dictionary_sugar_get(dict = mlr3torch_callbacks)
}

#' @title Convert to a TorchCallback
#'
#' @description
#' Converts an object to a [`TorchCallback`].
#'
#' @param x (any)\cr
#'   Object to be converted.
#' @param clone (`logical(1)`)\cr
#'   Whether to make a deep clone.
#' @param ... (any)\cr
#'   Additional arguments
#'
#' @return [`TorchCallback`].
#' @family Callback

#' @export
as_torch_callback = function(x, clone = FALSE, ...) {
  assert_flag(clone)
  UseMethod("as_torch_callback")
}

#' @export
as_torch_callback.TorchCallback = function(x, clone = FALSE, ...) { # nolint
  if (clone) x$clone(deep = TRUE) else x
}

#' @export
as_torch_callback.R6ClassGenerator = function(x, clone = FALSE, id = deparse(substitute(x))[[1L]], ...) { # nolint
  TorchCallback$new(callback_generator = x, id = id, ...)
}

#' @export
as_torch_callback.character = function(x, clone = FALSE, ...) { # nolint
  t_clbk(.key = x, ...)
}

#' @title Convert to a list of Torch Callbacks
#' @description
#' Converts an object to a list of [`TorchCallback`].
#'
#' @param x (any)\cr
#'   Object to convert.
#' @param clone (`logical(1)`)\cr
#'   Whether to create a deep clone.
#' @param ... (any)\cr
#'   Additional arguments.
#'
#' @family Callback
#' @family Torch Descriptor
#' @return `list()` of [`TorchCallback`]s
#' @export
as_torch_callbacks = function(x, clone, ...) {
  UseMethod("as_torch_callbacks")
}

#' @export
as_torch_callbacks.list = function(x, clone = FALSE, ...) { # nolint
  lapply(x, as_torch_callback, clone = clone, ...)
}

#' @export
as_torch_callbacks.NULL = function(x, clone = FALSE, ...) { # nolint
  list()
}

#' @export
as_torch_callbacks.default = function(x, clone = FALSE, ...) { # nolint
  list(as_torch_callback(x, clone = clone, ...))
}

#' @export
as_torch_callbacks.character = function(x, clone = FALSE, ...) { # nolint
  t_clbks(x, ...)
}

#' @title Torch Callback
#'
#' @description
#' This wraps a [`CallbackSet`] and annotates it with metadata, most importantly a [`ParamSet`][paradox::ParamSet].
#' The callback is created for the given parameter values by calling the `$generate()` method.
#'
#' This class is usually used to configure the callback of a torch learner, e.g. when constructing
#' a learner of in a [`ModelDescriptor`].
#'
#' For a list of available callbacks, see [`mlr3torch_callbacks`].
#' To conveniently retrieve a [`TorchCallback`], use [`t_clbk()`].
#'
#' @section Parameters:
#' Defined by the constructor argument `param_set`.
#' If no parameter set is provided during construction, the parameter set is constructed by creating a parameter
#' for each argument of the wrapped loss function, where the parametes are then of type `ParamUty`.
#'
#' @family Callback
#' @family Torch Descriptor
#'
#' @export
#' @examplesIf torch::torch_is_installed()
#' # Create a new torch callback from an existing callback set
#' torch_callback = TorchCallback$new(CallbackSetCheckpoint)
#' # The parameters are inferred
#' torch_callback$param_set
#'
#' # Retrieve a torch callback from the dictionary
#' torch_callback = t_clbk("checkpoint",
#'   path = tempfile(), freq = 1
#' )
#' torch_callback
#' torch_callback$label
#' torch_callback$id
#'
#' # open the help page of the wrapped callback set
#' # torch_callback$help()
#'
#' # Create the callback set
#' callback = torch_callback$generate()
#' callback
#' # is the same as
#' CallbackSetCheckpoint$new(
#'   path = tempfile(), freq = 1
#' )
#'
#' # Use in a learner
#' learner = lrn("regr.mlp", callbacks = t_clbk("checkpoint"))
#' # the parameters of the callback are added to the learner's parameter set
#' learner$param_set
#'
TorchCallback = R6Class("TorchCallback",
  inherit = TorchDescriptor,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    #' @param callback_generator (`R6ClassGenerator`)\cr
    #'   The class generator for the callback that is being wrapped.
    #' @template param_id
    #' @param param_set (`ParamSet` or `NULL`)\cr
    #'   The parameter set. If `NULL` (default) it is inferred from `callback_generator`.
    #' @template param_label
    #' @template param_packages
    #' @template param_man
    initialize = function(callback_generator, param_set = NULL, id = NULL,
      label = NULL, packages = NULL, man = NULL) {
      assert_class(callback_generator, "R6ClassGenerator")

      param_set = assert_param_set(param_set %??% inferps(callback_generator))
      if ("ctx" %in% param_set$ids()) {
        stopf("The name 'ctx' is reserved for the ContextTorch and cannot be a construction argument.")
      }
      super$initialize(
        generator = callback_generator,
        id = id,
        param_set = param_set,
        packages = union(packages, "mlr3torch"),
        label = label,
        man = man
      )
    }
  ),
  private = list(
    .additional_phash_input = function() NULL
  )
)

#' @title Create a Callback Desctiptor
#'
#' @description
#' Convenience function to create a custom [`TorchCallback`].
#' All arguments that are available in [`callback_set()`] are also available here.
#' For more information on how to correctly implement a new callback, see [`CallbackSet`].
#'
#' @inheritParams callback_set
#' @param id (`character(1)`)\cr`\cr
#'   The id for the torch callback.
#' @param param_set (`ParamSet`)\cr
#'   The parameter set, if not present it is inferred from the `$initialize()` method.
#' @param packages (`character()`)\cr`
#'   The packages the callback depends on. Default is `NULL`.
#' @param label (`character(1)`)\cr
#'   The label for the torch callback.
#'   Defaults to the capitalized `id`.
#' @param man (`character(1)`)\cr
#'   String in the format `[pkg]::[topic]` pointing to a manual page for this object.
#'   The referenced help package can be opened via method `$help()`.
#'   The default is `NULL`.
#'
#' @inheritSection mlr_callback_set Stages
#'
#' @section Internals:
#' It first creates an `R6` class inheriting from [`CallbackSet`] (using [`callback_set()`]) and
#' then wraps this generator in a [`TorchCallback`] that can be passed to a torch learner.
#'
#' @export
#' @return [`TorchCallback`]
#' @include zzz.R CallbackSet.R
#' @family Callback
#' @examplesIf torch::torch_is_installed()
#' custom_tcb = torch_callback("custom",
#'   initialize = function(name) {
#'     self$name = name
#'   },
#'   on_begin = function() {
#'     cat("Hello", self$name, ", we will train for ", self$ctx$total_epochs, "epochs.\n")
#'   },
#'   on_end = function() {
#'     cat("Training is done.")
#'   }
#' )
#'
#' learner = lrn("classif.torch_featureless",
#'   batch_size = 16,
#'   epochs = 1,
#'   callbacks = custom_tcb,
#'   cb.custom.name = "Marie",
#'   device = "cpu"
#' )
#' task = tsk("iris")
#' learner$train(task)
torch_callback = function(
  id,
  classname = paste0("CallbackSet", capitalize(id)),
  param_set = NULL,
  packages = NULL,
  label = capitalize(id),
  man = NULL,
  on_begin = NULL,
  on_end = NULL,
  on_exit = NULL,
  on_epoch_begin = NULL,
  on_before_valid = NULL,
  on_epoch_end = NULL,
  on_batch_begin = NULL,
  on_batch_end = NULL,
  on_after_backward = NULL,
  on_batch_valid_begin = NULL,
  on_batch_valid_end = NULL,
  on_valid_end = NULL,
  state_dict = NULL,
  load_state_dict = NULL,
  # other arguments
  initialize = NULL,
  public = NULL, private = NULL, active = NULL, parent_env = parent.frame(), inherit = CallbackSet,
  lock_objects = FALSE
  ) {

  callback_generator = callback_set(
    classname = classname,
    # training
    on_begin = on_begin,
    on_end = on_end,
    on_exit = on_exit,
    on_epoch_begin = on_epoch_begin,
    on_before_valid = on_before_valid,
    on_epoch_end = on_epoch_end,
    on_batch_begin = on_batch_begin,
    on_batch_end = on_batch_end,
    on_after_backward = on_after_backward,
    # validation
    on_batch_valid_begin = on_batch_valid_begin,
    on_batch_valid_end = on_batch_valid_end,
    on_valid_end = on_valid_end,
    # other arguments
    state_dict = state_dict, load_state_dict = load_state_dict,
    initialize = initialize,
    public = public, private = private, active = active, parent_env = parent_env, inherit = inherit,
    lock_objects = lock_objects
  )

  TorchCallback$new(
    callback_generator = callback_generator,
    param_set = param_set,
    packages = packages,
    id = id,
    man = man,
    label = label
  )
}

#' @title Dictionary of Torch Callbacks
#'
#' @description
#' A [`mlr3misc::Dictionary`] of torch callbacks.
#' Use [`t_clbk()`] to conveniently retrieve callbacks.
#' Can be converted to a [`data.table`][data.table::data.table] using
#' [`as.data.table`][data.table::as.data.table].
#'
#' @family Callback
#' @family Dictionary
#' @export
#' @examplesIf torch::torch_is_installed()
#' mlr3torch_callbacks$get("checkpoint")
#' # is the same as
#' t_clbk("checkpoint")
#' # convert to a data.table
#' as.data.table(mlr3torch_callbacks)
mlr3torch_callbacks = R6Class("DictionaryMlr3torchCallbacks",
  inherit = Dictionary,
  cloneable = FALSE
)$new()


#' @export
as.data.table.DictionaryMlr3torchCallbacks = function(x, ...) { # nolint
  setkeyv(map_dtr(x$keys(), function(key) {
    cb = x$get(key)
    list(
      key = key,
      label = cb$label,
      packages = paste0(cb$packages, collapse = ",")
    )
  }), "key")[]
}
