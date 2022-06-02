#' @title Dictionary with Torch Callbackss
#' @description
#' Dictionary containing callbacks for the torch Learner.
#' @export
torch_callbacks = mlr3misc::Dictionary$new()

#' @title Retrieve a callback
#' @param id (`character(1)`)\cr
#'   The callback's id.
#' @param ... (any)\cr
#'   Additional initialization arguments for the callback.
#' @export
cllb = function(id, ...) {
  torch_callbacks$get(id, ...)
}

#' @title Retrieve callbacks
#' @param ids (`character(1)`)\cr
#'   The callback's id.
#' @param ... (any)\cr
#'   Additional initialization arguments for the callbacks.
#' @export
cllbs = function(ids, ...) {
  map(ids, function(id) cllb(id, ...))
}


#' @title Callbacks for Torch Learner
#' @description
#' All Callbacks for the Torch Learners should inherit from this class.
#' @export
CallbackTorch = R6Class("CallbackTorch",
  inherit = Callback,
  public = list(
    #' @description Initializes an instance of this [R6][R6::R6Class] class.
    #' @param id (`character(1)`)\cr
    #'   The id for the new object.
    initialize = function(id) {
      x = names(self)
      callbacks = x[startsWith(x, "on_")]
      steps = gsub("on_", "", callbacks)
      invalid_steps = steps %nin% torch_reflections$callback_steps
      if (sum(invalid_steps > 0L)) {
        message = paste0(
          "Invalid public method(s) ", paste(callbacks[invalid_steps], collapse = ", "), "."
        )
        stopf(message)
      }
      super$initialize(id)
    }
  )
)
