#' @title Callback
#'
#' @description
#' Callback
#'
#' Use `callback()` function to create a [Callback].
#'
#' @export
Callback = R6::R6Class("Callback",
  lock_objects = FALSE,
  public = list(
    #' @field id (`character(1)`)\cr
    #'   Identifier for the callback.
    id = NULL,

    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    #'
    #' @param id (`character(1)`)\cr
    #'   Identifier for the new callback.
    initialize = function(id) {
      self$id = assert_character(id)
    },

    #' @description
    #' Call.
    #'
    #' @param step (`character(1)`)\cr
    #'   Step.
    call = function(step) {
      if (!is.null(self[[step]])) {
        self[[step]]()
      }

    },
    #' @field context ([`Context`][Context])\cr
    #'   The context in which to call the callback.
    context = NULL
  )
)

#' @title Create a Callback
#'
#' @description
#' Create a [Callback].
#'
#' @param id (`character(1)`)\cr
#'   Identifier for the new callback.
#'
#' @param ... (Named list of `function()`s)
#'   Public methods of the [Callback].
#'   The functions must have a single argument named `context`.
#'   The argument names indicate the step in which the method is called.
as_callback = function(id, ...) {
  callback_methods = list(...)
  assert_subset(names(callback_methods), bbotk_reflections$callback_steps)
  walk(callback_methods, function(method) assert_names(formalArgs(method), identical.to = "context"))
  callback = Callback$new(id = id)
  iwalk(callback_methods, function(method, step) {
    callback[[step]] = method
  })
  callback
}

#' @title Call Callbacks
#'
#' @description
#' Call list of callbacks with context at specific step.
#'
#' @keywords internal
#' @export
call_back = function(step, callbacks) {
  callbacks = Filter(function(x) !is.null(x), callbacks)
  walk(callbacks, function(callback) callback$call(step))
  return(invisible(NULL))
}
