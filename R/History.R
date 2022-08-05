#' @title Training History for a Torch Learner
#'
#' @description
#' Training History for a Torch Learner
#' The train logs are saved in the field '$train` and the validation logs are saved in the
#' field `$valid`. They are named lists respectively, each containing a list again that contains
#' a list for each epoch.
#'
#' @export
History = R6Class("History",
  lock_objects = FALSE,
  public = list(
    #' @field train (`list()`)\cr
    #'   Contains the training losses.
    train_loss = list(),
    #' @field valid (`list()`)\cr
    #' A list containing the different measures evaluated for each validation epoch.
    valid = NULL,
    #' @field train (`list()`)\cr
    #' A list containing the different measures evaluated for each training epoch.
    train = NULL,
    #' @description Initializes an object of this [R6][R6::R6Class] class.
    #' @param measures_valid (`list()`)\cr
    #'   List of measures to calculate as validation metrics.
    #' @param measures_valid (`list()`)\cr
    #'   List of measures to calculate as training metrics.
    initialize = function(measures_valid, measures_train) {
      self$valid = named_list(nn = map_chr(measures_valid, "id"), init = list())
      self$train = named_list(nn = map_chr(measures_train, "id"), init = list())
      self$train_loss = list()
    }
  )
)
