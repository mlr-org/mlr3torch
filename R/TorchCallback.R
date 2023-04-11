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
#' @return [`CallbackTorch`]
#'
#' @export
#' @family callback
#' @examples
#' t_clbk("progress")
t_clbk = function(.key, ...) {
  dictionary_sugar_get(dict = mlr3torch_callbacks, .key = .key, ...)
}


#' @rdname t_clbk
#' @param .keys (`character()`)\cr
#'   The keys of the callbacks.
#' @export
t_clbks = function(.keys, ...) {
  dictionary_sugar_mget(dict = mlr3torch_callbacks, .keys = .keys, ...)
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
as_torch_callback.R6ClassGenerator = function(x, clone = FALSE, param_set = NULL, ...) { # nolint
  # no need to clone because we have a class generator
  # TorchCallback$new(callback_generator = x, param_set = param_set$clone(deep = TRUE), ...)
  if (!is.null(param_set)) {
    assert_param_set(param_set)
    param_set = if (clone) param_set$clone(deep = TRUE) else param_set
  }
  TorchCallback$new(callback_generator = x, param_set = param_set)
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
as_torch_callbacks.default = function(x, clone = FALSE, ...) { # nolint
  list(as_torch_callback(x, clone = clone, ...))
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
#' @family torch_wrappers, callback
#' @include utils.R
#' @export
TorchCallback = R6Class("TorchCallback",
  inherit = TorchWrapper,
  public = list(
    man = NULL,
    initialize = function(callback_generator, param_set = NULL, id = deparse(substitute(callback_generator))[[1]], 
      label = id, packages = NULL, man = NULL) {
      self$man = assert_string(man, min.chars = 1, null.ok = TRUE)
      super$initialize(
        generator = callback_generator,
        id = id,
        param_set = param_set,
        packages = packages,
        label = label
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
#' @param id (`character(1)`)\cr`\cr
#'   The id for the callbacks.
#'   Note that the ids of callbacks passed to a learner must be unique.
#' @param name (`character(1)`)\cr
#'   The class name of the torch callback. Is set to `"CallbackTorch<Id>"` per default.
#'   E.g. id `id` is `"custom"`, the name is set to `"CallbackTorchCustom"`.
#' @param param_set (`ParamSet`)\cr
#'   The parameter set, if not present it is inferred from the initialize method passed through the public function.
#' @param packages (`character()`)\cr`
#'   The packages the callback depends on. Default is `NULL`.
#' @param label (`character(1)`)\cr
#'   Label for the new instance.
#' @param man (`character(1)`)\cr
#'   String in the format `[pkg]::[topic]` pointing to a manual page for this object.
#'   The referenced help package can be opened via method `$help()`.
#' @param on_begin, on_end, on_epoch_begin, on_before_valid, on_epoch_end, on_batch_begin, on_batch_end,
#' on_after_backward, on_batch_valid_begin, on_batch_valid_end (`function`)\cr
#' Function to execute at the given stage, see section *Stages*.
#' @param public (`list()`)\cr
#'   Additional public fields to add to the callback.
#' @param private (`list()`)\cr
#'   Additional private fields to add to the callback.
#' @param active (`list()`)\cr
#'   Additional active fields to add to the callback.
#' @param parent_env (`environment()`)\cr
#'   The parent environment for the [`R6Class`].
#'
#' @inheritSection mlr_callbacks_torch Stages
#'
#'
#' @section Internals:
#' It first creates an [`R6ClassGenerator`] that generates a [`CallbackTorch`] and when wraps this generator in a
#' [`TorchCallback`].
#'
#' @export
#' @return [`TorchCallback`]
#' @include zzz.R
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
#' ctx$name = "Julia"
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
  name = paste0("CallbackTorch", capitalize(id)),
  param_set = NULL,
  packages = NULL,
  label = id,
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
  # prediction
  public = NULL, private = NULL, active = NULL, parent_env = parent.frame()) {
  assert_string(id, min.chars = 1L)
  more_public = list(
    on_begin = assert_function(on_begin, args = "ctx", null.ok = TRUE),
    on_end = assert_function(on_end, args = "ctx", null.ok = TRUE),
    on_epoch_begin = assert_function(on_epoch_begin, args = "ctx", null.ok = TRUE),
    on_before_valid = assert_function(on_before_valid, args = "ctx", null.ok = TRUE),
    on_epoch_end = assert_function(on_epoch_end, args = "ctx", null.ok = TRUE),
    on_batch_begin = assert_function(on_batch_begin, args = "ctx", null.ok = TRUE),
    on_batch_end = assert_function(on_batch_end, args = "ctx", null.ok = TRUE),
    on_after_backward = assert_function(on_after_backward, args = "ctx", null.ok = TRUE),
    on_batch_valid_begin = assert_function(on_batch_valid_begin, args = "ctx", null.ok = TRUE),
    on_batch_valid_end = assert_function(on_batch_valid_end, args = "ctx", null.ok = TRUE),
    id = id
  )

  assert_list(public, null.ok = TRUE, names = "unique")
  if (length(public)) assert_names(names(public), disjunct.from = names(more_public))

  invalid_stages = names(public)[grepl("^on_", names(public))]

  if (length(invalid_stages)) {
    warningf("There are public method(s) with name(s) %s, which are not valid stages.",
      paste(paste0("'", invalid_stages, "'"), collapse = ", ")
    )
  }
  assert_list(private, null.ok = TRUE, names = "unique")
  assert_list(active, null.ok = TRUE, names = "unique")
  assert_environment(parent_env)

  more_public = Filter(function(x) !is.null(x), more_public)
  parent_env_shim = new.env(parent = parent_env)
  parent_env_shim$inherit = CallbackTorch
  callback_generator = R6::R6Class(classname = name, inherit = CallbackTorch, public = c(more_public, public),
    private = private, active = active, parent_env = parent_env_shim, lock_objects = FALSE)

  TorchCallback$new(
    callback_generator = callback_generator,
    param_set = param_set,
    packages = packages,
    id = id,
    man = man,
    label = label
    
  )
}
