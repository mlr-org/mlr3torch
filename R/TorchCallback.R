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
#' @family callback, torch_wrapper
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
#' @description
#' Converts an object to a [`TorchCallback`].
#'
#' @param x (any)\cr
#'   Object to be converted.
#' @param clone (`logical(1)`\cr
#'   Whether to make a deep clone.
#' @param ... (any)\cr
#'   Additional arguments
#' @param param_set ([`ParamSet`])\cr
#'   The parameter set.
#'
#' @return [`TorchCallback`].
#' @family callback

#' @export
as_torch_callback = function(x, clone = FALSE, ...) {
  assert_flag(clone)
  UseMethod("as_torch_callback")
}

#' @export
as_torch_callback.TorchCallback = function(x, clone = FALSE) { # nolint
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
#' @family callback
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

as_torch_callbacks.character = function(x, clone = FALSE, ...) { # nolint
  t_clbks(x, ...)
}

#' @title Torch Callback
#'
#' @usage NULL
#' @name TorchCallback
#' @format `r roxy_format(TorchCallback)`
#'
#' @description
#' Leight-weight wrapper around callback for torch: A [`TorchCallback`] wraps a [`CallbackTorch`].
#' To conveniently retrieve a [`TorchCallback`], use [`t_clbk`].
#' It is an analogous construct to the classes [`TorchOptimizer`] or [`TorchLoss`] which wrap torch optimizers and
#' losses.
#'
#' @section Construction: `r roxy_construction(TorchCallback)`
#' Arguments from [`TorchWrapper`] (except for `generator`) as well as:
#' * `callback_generator` :: [`R6ClassGenerator`]\cr
#'   The class generator for the callback that is being wrapped.
#'
#' @section Parameters:
#' Defined by the constructor argument `param_set`.
#'
#' @section Fields:
#' Only fields inherited from [`TorchWrapper`] as well as:
#'
#' @section Methods:
#' Only methods inherited from [`TorchWrapper`].
#'
#' @family callback, model_configuration, torch_wrapper
#' @include utils.R
#' @export
TorchCallback = R6Class("TorchCallback",
  inherit = TorchWrapper,
  public = list(
    man = NULL,
    initialize = function(callback_generator, param_set = NULL, id = deparse(substitute(id))[[1L]],
      label = capitalize(id), packages = NULL, man = NULL) {
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
#' It returns a [`TorchCallback`] that wrapping a [`CallbackTorch`].
#'
#'
#' @inheritParams callback_torch
#' @param id (`character(1)`)\cr`\cr
#'   The id for the callbacks.
#'   Note that the ids of callbacks passed to a learner must be unique.
#' @param param_set (`ParamSet`)\cr
#'   The parameter set, if not present it is inferred from the initialize method passed through the public function.
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
#'
#' @section Internals:
#' It first creates an [`R6ClassGenerator`] using [`torch_callback`]and when wraps this generator in a
#' [`TorchCallback`].
#'
#' @export
#' @return [`TorchCallback`]
#' @include zzz.R CallbackTorch.R
#' @family callback
#' @examples
#' custom_tcb = torch_callback(
#'   "mycallback",
#'   public = list(
#'     initialize = function(greeting) {
#'       self$greeting = greeting
#'     }
#'   ),
#'   on_begin = function(ctx) {
#'     cat(self$greeting, ctx$name, "\n")
#'   },
#'   on_end = function(ctx) {
#'     cat("Bye", ctx$name, "\n")
#'   }
#' )
#'
#' custom_tcb$param_set$set_values(greeting = "Wazzuuup")
#'
#' cb = custom_tcb$get_callback()
#'
#' ctx = new.env()
#' ctx$name = "Albert"
#'
#' f = function(ctx, cb) {
#'   cb$on_begin(ctx)
#'   catn("Doing heavy work ...")
#'   cb$on_end(ctx)
#' }
#'
#' f(ctx, cb)
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
  public = NULL, private = NULL, active = NULL, parent_env = parent.frame()) {

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
    public = public, private = private, active = active, parent_env = parent_env
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
#' @usage NULL
#' @format [`R6Class`] inheriting from [`Dictionary`].
#'
#' @description
#' A [`mlr3misc::Dictionary`] of torch callbacks.
#' Use [`t_clbk`] to conveniently retrieve callbacks.
#' Can be converted to a [`data.table`] using `as.data.table`.
#'
#' @section Methods:
#' See [mlr3misc::Dictionary].
#'
#' @family callback
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
