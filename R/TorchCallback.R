#' @title Sugar Function to Retrieve Torch Callback(s)
#'
#' @description
#' Retrieves one or more torch callback from [`mlr3torch_callbacks`].
#' Works like [`mlr3::lrn()`] or [`mlr3::tsk()`].
#'
#' @param .key (`character(1)`)\cr
#'   The key of the callback.
#' @param ... (any)\cr
#'   See description of [`dictionary_sugar_get`].
#'
#' @return [`TorchCallback`]
#'
#' @export
#' @family Callback
#' @family Descriptor Torch
#' @examples
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
#' @param clone (`logical(1)`\cr
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
#' @family Descriptor Torch
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
#' This wraps a [`CallbackTorch`] and annotates it with metadata, most importantly a [`ParamSet`].
#' The callback is created for the given parameter values by calling the `$generate()` method.
#'
#' This class is usually used to configure the callback of a torch learner, e.g. when constructing
#' a learner of in a [`ModelDescriptor`].
#'
#' For a list of available callbacks, see mlr3torch_callbacks
#' To conveniently retrieve a [`TorchCallback`], use [`t_clbk()`].
#'
#' @section Parameters:
#' Defined by the constructor argument `param_set`.
#' If no parameter set is provided during construction, the parameter set is constructed by creating a parameter
#' for each argument of the wrapped loss function, where the parametes are then of type [`ParamUty`].
#'
#' @family Callback
#' @family Descriptor Torch
#'
#' @export
#' @examples
#' # Create a new Torch Callback from an existing callback
#' torchcallback = TorchCallback$new(CallbackTorchCheckpoint)
#' # The parameters are inferred
#' torchcallback$param_set
#'
#' # Retrieve a torch callbac from the dictionary
#' torchcallback = t_clbk("checkpoint",
#'   path = tempfile(), freq = 1
#' )
#' torchcallback
#' torchcallback$label
#' torchcallback$id
#'
#' # Create the callback
#' callback = torchcallback$generate()
#' callback
#' # is the same as
#' CallbackTorchCheckpoint$new(
#'   path = tempfile(), freq = 1
#' )
#'
#' # open the help page of the wrapped callback
#' # torchcallback$help()
#'
#' # Use in a learner
#' learner = lrn("regr.mlp", callbacks = t_clbk("checkpoint"))
#' # the parameters of the callback are added to the learner's parameter set
#' learner$param_set
#'
TorchCallback = R6Class("TorchCallback",
  inherit = DescriptorTorch,
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
    initialize = function(callback_generator, param_set = NULL, id = deparse(substitute(id))[[1L]],
      label = capitalize(id), packages = NULL, man = NULL) {
      assert_class(callback_generator, "R6ClassGenerator")

      param_set = assert_param_set(param_set %??% inferps(callback_generator))
      if ("ctx" %in% param_set$ids()) {
        stopf("The name 'ctx' is reserved for the ContextTorch and cannot be a construction argument.")
      }
      super$initialize(
        generator = callback_generator,
        id = id,
        param_set = param_set,
        packages = packages,
        label = label,
        man = man
      )
    }
  )
)

#' @title Create a Torch Callback
#'
#' @description
#' Convenience function to create a custom callback for torch.
#' For more information on how to correctly implement a new callback, see [`CallbackTorch`].
#' It returns a [`TorchCallback`] wrapping a [`CallbackTorch`].
#'
#' @inheritParams callback_torch
#' @param id (`character(1)`)\cr`\cr
#'   The id for the callbacks.
#'   Note that the ids of callbacks passed to a learner must be unique.
#' @param param_set (`ParamSet`)\cr
#'   The parameter set, if not present it is inferred from the initialize method.
#' @param packages (`character()`)\cr`
#'   The packages the callback depends on. Default is `NULL`.
#' @param label (`character(1)`)\cr
#'   Label for the new instance.
#' @param man (`character(1)`)\cr
#'   String in the format `[pkg]::[topic]` pointing to a manual page for this object.
#'   The referenced help package can be opened via method `$help()`.
#'
#' @inheritSection mlr_callbacks_torch Stages
#'
#' @section Internals:
#' It first creates an `R6ClassGenerator` using [`torch_callback`] and when wraps this generator in a
#' [`TorchCallback`].
#'
#' @export
#' @return [`TorchCallback`]
#' @include zzz.R CallbackTorch.R
#' @family Callback
#' @examples
#' custom_tcb = torch_callback("custom",
#'   initialize = function(name) {
#'     self$name = name
#'   },
#'   on_begin = function {
#'     cat("Hello", self$name, ", we will train for ", self$ctx$total_epochs, "epochs.\n")
#'   },
#'   on_end = function {
#'     cat("Training is done.")
#'   }
#' )
#'
#' learner = lrn("classif.torch_featureless",
#'   batch_size = 16,
#'   epochs = 1,
#'   callbacks = custom_tcb,
#'   cb.custom.name = "Marie"
#' )
#' task = tsk("iris")
#' learner$train(task)
torch_callback = function(
  id,
  classname = paste0("CallbackTorch", capitalize(id)),
  param_set = NULL,
  packages = NULL,
  label = capitalize(id),
  man = NULL,
  # training
  on_begin = NULL,
  on_end = NULL,
  on_epoch_begin = NULL,
  on_before_valid = NULL,
  on_epoch_end = NULL,
  on_batch_begin = NULL,
  on_batch_end = NULL,
  on_after_backward = NULL,
  # validation
  on_batch_valid_begin = NULL,
  on_batch_valid_end = NULL,
  # other arguments
  initialize = NULL,
  public = NULL, private = NULL, active = NULL, parent_env = parent.frame(), inherit = CallbackTorch
  ) {

  callback_generator = callback_torch(
    classname = classname,
    # training
    on_begin = on_begin,
    on_end = on_end,
    on_epoch_begin = on_epoch_begin,
    on_before_valid = on_before_valid,
    on_epoch_end = on_epoch_end,
    on_batch_begin = on_batch_begin,
    on_batch_end = on_batch_end,
    on_after_backward = on_after_backward,
    # validation
    on_batch_valid_begin = on_batch_valid_begin,
    on_batch_valid_end = on_batch_valid_end,
    # other arguments
    initialize = initialize,
    public = public, private = private, active = active, parent_env = parent_env, inherit = inherit
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
#' Can be converted to a [`data.table`] using `as.data.table`.
#'
#' @family Callback
#' @family Dictionary
#' @export
#' @examples
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
