#' @title Base Class for Torch Callbacks
#'
#' @name mlr_callbacks_torch
#' @format `r roxy_format(CallbackTorch)`
#'
#' @description
#' Base class for Torch Callbacks.
#' To create custom callbacks,to use in a torch learner use [`torch_callback`].
#'
#' Torch Callbacks can be used to gain more control over the training process of a neural network without
#' having to write the whole training loop.
#' At each stage (see section "Stages") of the training loop, the corresponding `on_<stage>(ctx)` method is run
#' that takes as argument a [`ContextTorch`] which gives access to the relevant objects.
#'
#'
#' @section Construction:
#' `r roxy_construction(CallbackTorch)`
#' See [`Callback`] for the construction arguments.
#'
#' @section Fields:
#' * `id` :: (`character(1)`)\cr
#'   The id of the callback.
#' * `label` :: (`character(1)`)\cr
#'   The label of the callback.
#'
#' @section Methods:
#' See section *Stages*.
#'
#' @section Inheriting:
#' It is recommended to use the sugar function [`callback_torch()] to create custom callbacks.
#'
#' @section Stages:
#' * `begin` :: Run before the training loop begins.
#' * `epoch_begin` :: Run he beginning of each epoch.
#' * `before_validation` :: Run before each validation loop.
#' * `batch_begin` :: Run before the forward call.
#' * `after_backward` :: Run after the backward call.
#' * `batch_end` :: Run after the optimizer step.
#' * `batch_valid_begin` :: Run before the forward call in the validation loop.
#' * `batch_valid_end` :: Run after the forward call in the validation loop.
#' * `epoch_end` :: Run at the end of each epoch.
#' * `end` :: Run at last, using `on.exit()`.
#'
#'
#' @export
CallbackTorch = R6Class("CallbackTorch",
  lock_objects = FALSE,
  cloneable = FALSE,
  public = list(
    id = NULL,
    label = NULL,
    on_begin = function(ctx) NULL,
    on_epoch_begin = function(ctx) NULL,
    on_before_validation = function(ctx) NULL,
    on_epoch_end = function(ctx) NULL,
    on_batch_begin = function(ctx) NULL,
    on_batch_end = function(ctx) NULL,
    on_after_backward = function(ctx) NULL,
    on_batch_valid_begin = function(ctx) NULL,
    on_batch_valid_end = function(ctx) NULL,
    on_end = function(ctx) NULL
  )
)

#' @title Callbacks
#' @description
#' Dictionary of torch callbacks.
#' Use [`t_clbk`] for conveniently retrieving callbacks.
#'
#' @section Available Callbacks:
#'
#' * history - [`CallbackTorchHistory`]
#' * progress - [`CallbackTorchProgress`]
#'
#' @export
mlr3torch_callbacks = R6Class("DictionaryMlr3torchCallbacks",
  inherit = Dictionary,
  cloneable = FALSE
)$new()

#' @export
mlr3misc::clbk

#' @export
mlr3misc::mlr_callbacks

#' Sugar Function to Retrieve Torch Callback(s)
#'
#' Retrieves one or more torch callback from the callback registry.
#'
#' @param .key, (`character(1)`)\cr
#'   The key of the callback.
#' @param .keys, (`character()`)\cr
#'   The keys of the callbacks.
#' @param ... (any)\cr
#'   See description of [`dictionary_sugar_get`].
#'
#' @return A [`CallbackTorch`]
#'
#' @export
#' @examples
#' t_clbk("progress")
#' # is the same as
#' clbk("torch.progress")
t_clbk = function(.key, ...) {
  dictionary_sugar_get(dict = mlr3torch_callbacks, .key = .key, ...)
}


#' @rdname t_clbk
#' @export
t_clbks = function(.keys, ...) {
  dictionary_sugar_mget(dict = mlr3torch_callbacks, .keys = .keys, ...)
}


#' @title Create a TorchCallback
#'
#' @description
#' Convenience function to create a custom callback for torch.
#' For more information on how to correctly implement a new callback, see [`CallbackTorch`].
#'
#' @section Internals:
#' It first creates an [`R6ClassGenerator`] that generates a [`CallbackTorch`] and when wraps this generator in a
#' [`TorchCallback`].
#'
#' @param id
#'
#' @param name (`character(1)`)\cr
#'   The class name, e.g. `"CallbackTorchCustom"`.
#'   Per default i
#' @param on_begin, on_end, on_epoch_begin, on_before_validation, on_epoch_end, on_batch_begin, on_batch_end,
#' on_after_backward, on_batch_valid_begin, on_batch_valid_end (`function( )\cr
#' Function to execute at the given stage, see section *Stages*.
#'
#' @param id (`character(1)`)\cr`\cr
#'   The id for the callbacks. Note that the ids of callbacks passted to a learner must be unique.
#' @param param_set (`ParamSet`)\cr
#'   The parameter set, if not present it is inferred from the initialize method.
#' @param name (`character(1)`)\cr
#'   The class name of the torch callback. Is set to `"CallbackTorch<Id>"` per default.
#'   E.g. id `id` is `"custom"`, the name is set to `"CallbackTorchCustom"`.
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
#' @include TorchCallback.R
#' @export
torch_callback = function(
  id,
  param_set = NULL,
  name = paste0("CallbackTorch", capitalize(id)),
  # training
  on_begin = NULL,
  on_end = NULL,
  on_epoch_begin = NULL,
  on_before_validation = NULL,
  on_epoch_end = NULL,
  on_batch_begin = NULL,
  on_batch_end = NULL,
  on_after_backward = NULL,
  # validation
  on_batch_valid_begin = NULL,
  on_batch_valid_end = NULL,
  # predcition
  public = NULL, private = NULL, active = NULL, parent_env = parent.frame()) {
  assert_string(id, min.chars = 1L)
  more_public = list(
    on_begin = assert_function(on_begin, args = "ctx", null.ok = TRUE),
    on_end = assert_function(on_end, args = "ctx", null.ok = TRUE),
    on_epoch_begin = assert_function(on_epoch_begin, args = "ctx", null.ok = TRUE),
    on_before_validation = assert_function(on_before_validation, args = "ctx", null.ok = TRUE),
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
  assert_list(private, null.ok = TRUE, names = "unique")
  assert_list(active, null.ok = TRUE, names = "unique")
  assert_environment(parent_env)
  inherit = CallbackTorch
  more_public = Filter(function(x) !is.null(x), more_public)
  parent_env_shim = new.env(parent = parent_env)
  parent_env_shim$inherit = inherit
  callback_generator = R6::R6Class(classname = name, inherit = inherit, public = c(more_public, public),
    private = private, active = active, parent_env = parent_env_shim, lock_objects = FALSE)

  TorchCallback$new(callback_generator = callback_generator, param_set = param_set)
}


