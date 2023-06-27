#' @title Base Class for Torch Callbacks
#'
#' @name mlr_callbacks_torch
#'
#' @description
#' Base class from which Callbacks should inherit.
#' They can be used to gain more control over the training process of a neural network without
#' having to write everything from scratch.
#'
#' For each available stage (see section *Stages*) a public method `$on_<stage>()` can be defined.
#' The evaluation context (a [`ContextTorch`]) can be accessed via `self$ctx`, which contains
#' the current state of the training loop.
#' This context is assigned at the beginning of the training loop and removed afterwards.
#' Different stages of a callback can communicate with each other by assigning values to `$self`.
#'
#' When used a in torch learner, the `CallbackTorch` is wrapped in a [`TorchCallback`].
#' The latters parameter set represents the arguments of the [`CallbackTorch`]'s `$initialize()` method and can
#' be specified in the learner. The callback is then initialized at the beginning of the training loop.
#'
#' For creating custom callbacks, the function [`torch_callback()`] is recommended, which creates a
#' [`CallbackTorch`] and then wraps it in a [`TorchCallback`].
#'
#' @section Inheriting:
#' When one inherits from this class and overwrites the private `$deep_clone()` method, it is
#' important to ensure that a deep clone can only be created if the `ctx` field is `NULL`,
#' as this should never be cloned.
#' Furthermore, it is convenient to set `lock_objects` to `FALSE` to be able to freely assign variables
#' to `self`.
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
#' @family Callback
#' @export
CallbackTorch = R6Class("CallbackTorch",
  lock_objects = FALSE,
  cloneable = FALSE,
  public = list(
    #' @field ctx ([`TorchContext`] or `NULL`)\cr
    #'   The evaluation context for the callback.
    #'   This field should always be `NULL` except during the `$train()` call of the torch learner.
    ctx = NULL
  ),
  private = list(
    deep_clone = function(name, value) {
      if (name == "ctx" && !is.null(value)) {
        stopf("CallbackTorch instances must never be cloned unless the ctx is NULL.")
      } else {
        value
      }
    }
  )
)

#' @title Create a Callback Torch
#'
#' @description
#' Creates an `R6ClassGenerator` inheriting from [`CallbackTorch`].
#' Additionally performs checks such as that the stages are not accidentally misspelled.
#' To create a [`TorchCallback`] use [`torch_callback()`].
#'
#' In order for the resulting class to be cloneable, the private method `$deep_clone()` must be
#' provided.
#'
#' @param classname (`character(1)`)\cr
#'   The class name.
#' @param on_begin,on_end,on_epoch_begin,on_before_valid,on_epoch_end,on_batch_begin,on_batch_end,on_after_backward,on_batch_valid_begin,on_batch_valid_end (`function`)\cr
#'   Function to execute at the given stage, see section *Stages*.
#' @param initialize (`function()`)\cr
#'   The initialization method of the callback.
#' @param public,private,active (`list()`)\cr
#'   Additional public, private, and active fields to add to the callback.
#' @param parent_env (`environment()`)\cr
#'   The parent environment for the [`R6Class`].
#' @param inherit (`R6ClassGenerator`)\cr
#'   From which class to inherit.
#'   This class must either be [`CallbackTorch`] (default) or inherit from it.
#' @family Callback
#'
#' @export
callback_torch = function(
  classname,
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
  # other methods
  initialize = NULL,
  public = NULL, private = NULL, active = NULL, parent_env = parent.frame(), inherit = CallbackTorch
  ) {
  assert_true(startsWith(classname, "CallbackTorch"))
  more_public = list(
    on_begin = assert_function(on_begin, nargs = 0, null.ok = TRUE),
    on_end = assert_function(on_end, nargs = 0, null.ok = TRUE),
    on_epoch_begin = assert_function(on_epoch_begin, nargs = 0, null.ok = TRUE),
    on_before_valid = assert_function(on_before_valid, nargs = 0, null.ok = TRUE),
    on_epoch_end = assert_function(on_epoch_end, nargs = 0, null.ok = TRUE),
    on_batch_begin = assert_function(on_batch_begin, nargs = 0, null.ok = TRUE),
    on_batch_end = assert_function(on_batch_end, nargs = 0, null.ok = TRUE),
    on_after_backward = assert_function(on_after_backward, nargs = 0, null.ok = TRUE),
    on_batch_valid_begin = assert_function(on_batch_valid_begin, nargs = 0, null.ok = TRUE),
    on_batch_valid_end = assert_function(on_batch_valid_end, nargs = 0, null.ok = TRUE)
  )

  assert_function(initialize, null.ok = TRUE)

  if (!is.null(initialize)) {
    assert_true("initialize" %nin% names(public))
    public$initialize = initialize
  }

  assert_list(public, null.ok = TRUE, names = "unique")
  if (length(public)) assert_names(names(public), disjunct.from = names(more_public))

  invalid_stages = names(public)[grepl("^on_", names(public))]

  if (length(invalid_stages)) {
    warningf("There are public method(s) with name(s) %s, which are not valid stages.",
      paste(paste0("'", invalid_stages, "'"), collapse = ", ")
    )
  }
  cloneable = test_function(private$deep_clone, args = c("name", "value"))

  assert_list(private, null.ok = TRUE, names = "unique")
  assert_list(active, null.ok = TRUE, names = "unique")
  assert_environment(parent_env)
  assert_inherits_classname(inherit, "CallbackTorch")

  more_public = Filter(function(x) !is.null(x), more_public)
  parent_env_shim = new.env(parent = parent_env)
  parent_env_shim$inherit = inherit
  R6::R6Class(classname = classname, inherit = inherit, public = c(public, more_public),
    private = private, active = active, parent_env = parent_env_shim, lock_objects = FALSE,
    cloneable = cloneable
  )
}
