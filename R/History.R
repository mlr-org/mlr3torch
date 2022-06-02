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
  public = list(
    #' @field train (`list()`)\cr
    #'   Contains the training history.
    train = list(),
    #' @field valid (`list()`)\cr
    #'   Contains the validation history.
    valid = list(),
    #' @field steps (`list()`)\cr
    #'   Contains the steps (number of batches) for training and validation, i.e.
    #'   `steps$train` and `steps$valid` are the train and validation steps respecitvely.
    steps = list(),
    #' @field step (named `list()`)\cr
    #'   Named list where `step$train` contains the current training step and `step$valid`
    #'   indicates the current valdiation step
    step = list(),
    #' @field epoch (`integer(1)`)\cr
    #'   The current epoch.
    epoch = NULL,
    #' @description Initializes an instance of this [R6][R6::R6Class] class.
    #' @param steps_train (`integer(1)`)\cr
    #'   The number of training steps.
    #' @param steps_valid (`integer(1)`)\cr
    #'   The number of validation steps.
    initialize = function(steps_train, steps_valid) {
      self$steps$train = steps_train
      self$steps$valid = steps_valid

      self$step = list(train = 1, valid = 1)
      self$epoch = 1L
    },
    #' @description Appends the value for a measure.
    #' @param measure (`character(1)`)\cr
    #'   The measure that is saved.
    #' @param value (`character(1)`)\cr
    #'   The value for the masure that is being saved.
    #' @param phase (`character(1`)\cr
    #'   The phase, either "train" or "valid".
    append = function(measure, value, phase) {
      iter = self$step[[phase]]
      steps = self$steps[[phase]]

      if (is.null(self[[phase]][[measure]])) {
        self[[phase]][[measure]] = list(vector("list", self$steps[[phase]]))
      } else if (length(self[[phase]][[measure]]) < self$epoch) {
        self[[phase]][[measure]][[self$epoch]] = vector("list", self$steps[[phase]])
      }
      self[[phase]][[measure]][[self$epoch]][[iter]] = value

      invisible(self)
    },
    #' @description
    #' Increments the epoch or train/valid step.
    #' @param what (`character(1)`)\cr
    #'   Must be epoch, train or valid.
    increment = function(what) {
      if (what == "epoch") {
        self$epoch = self$epoch + 1L
        self$step$train = 1L
        self$step$valid = 1L
      } else if (what %in% c("train", "valid")) {
        self$step[[what]] = self$step[[what]] + 1L
      } else {
        stopf("Invalid x.")
      }
      invisible(self)
    }
  )
)
