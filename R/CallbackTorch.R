#' @title Base Class for Torch Callbacks
#'
#' @usage NULL
#' @name mlr_callbacks_torch
#' @format `r roxy_format(CallbackTorch)`
#'
#' @description
#' Base class from which Torch Callbacks should inherit.
#' To create custom callbacks,to use in a torch learner use the convenience function [`torch_callback`].
#'
#' Torch Callbacks can be used to gain more control over the training process of a neural network without
#' having to write everything from scratch.
#' At each stage (see section "Stages") of the training loop, the corresponding `on_<stage>(ctx)` method is run
#' that takes as argument a [`ContextTorch`] which gives access to the relevant objects.
#'
#' @section Construction:
#' `r roxy_construction(CallbackTorch)`
#'
#' @section Methods:
#' See section *Stages*.
#' Other methods can be added freely as well.
#'
#' @section Inheriting:
#' It is recommended to use the sugar function [`callback_torch()`] to create custom callbacks.
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
#' @family callback
#' @export
CallbackTorch = R6Class("CallbackTorch",
  lock_objects = FALSE,
  cloneable = FALSE,
)

#' @title Dictionary of Torch Callbacks
#'
#' @usage NULL
#' @format [R6::R6Class] object inheriting from [mlr3misc::Dictionary].
#'
#' @description
#' A [`mlr3misc::Dictionary`] of torch callbacks.
#' Use [`t_clbk`] to conveniently retrieve callbacks.
#'
#'
#' @section Methods:
#' See [mlr3misc::Dictionary].
#'
#' @family callback
#' @export
#' @examples
#' as.data.table(mlr3torch_callbacks)
#' mlr3torch_callbacks$get("checkpoint")
#' t_clbk("checkpoint")
mlr3torch_callbacks = R6Class("DictionaryMlr3torchCallbacks",
  inherit = Dictionary,
  cloneable = FALSE
)$new()
