#' @title Callback
#'
#' @description
#' Callback
#'
#' Use `callback()` function to create a [Callback].
#'
#' @export
CallbackTorch = R6::R6Class("CallbackTorch",
  lock_objects = FALSE,
  cloneable = FALSE,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    #'
    #' @param id (`character(1)`)\cr
    #'   Identifier for the new callback.
    initialize = function(state) {
      self$state = assert_r6(state, "TorchState")
    },
    on_begin = function() {},
    on_end = function() {},
    on_epoch_begin = function() {},
    on_before_validation = function() {},
    on_epoch_end = function() {},
    on_batch_begin = function() {},
    on_batch_end = function() {},
    on_after_backward = function() {},
    on_batch_valid_begin = function() {},
    on_batch_valid_end = function() {}
  )
)

#' @export
callback_torch = function(name = NULL, inherit = CallbackTorch,
    on_begin = NULL,
    on_end = NULL,
    on_epoch_begin = NULL,
    on_before_validation = NULL,
    on_epoch_end = NULL,
    on_batch_begin = NULL,
    on_batch_end = NULL,
    on_after_backward = NULL,
    on_batch_valid_begin = NULL,
    on_batch_valid_end = NULL,
    public = NULL, private = NULL, active = NULL, parent_env = parent.frame()) {
  more_public = list(
    on_begin = assert_function(on_begin, nargs = 0, null.ok = TRUE),
    on_end = assert_function(on_end, nargs = 0, null.ok = TRUE),
    on_epoch_begin = assert_function(on_epoch_begin, nargs = 0, null.ok = TRUE),
    on_before_validation = assert_function(on_before_validation, nargs = 0, null.ok = TRUE),
    on_epoch_end = assert_function(on_epoch_end, nargs = 0, null.ok = TRUE),
    on_batch_begin = assert_function(on_batch_begin, nargs = 0, null.ok = TRUE),
    on_batch_end = assert_function(on_batch_end, nargs = 0, null.ok = TRUE),
    on_after_backward = assert_function(on_after_backward, nargs = 0, null.ok = TRUE)
    on_batch_valid_begin = assert_function(on_batch_valid_begin, nargs = 0, null.ok = TRUE),
    on_batch_valid_end = assert_function(on_batch_valid_end, nargs = 0, null.ok = TRUE),
  )
  assert_list(public, null.ok = TRUE, names = "unique")
  assert_names(names(public), disjunct.from = names(more_public))
  assert_list(private, null.ok = TRUE, names = "unique")
  assert_list(active, null.ok = TRUE, names = "unique")
  assert_environment(parent_env)
  more_public = Filter(function(x) !is.null(x), more_public)
  R6::R6Class(classname = name, inherit = inherit, public = c(more_public, public),
    private = private, active = active, parent_env = parent_env, lock_objects = FALSE)
}

