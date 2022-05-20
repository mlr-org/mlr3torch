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
    train = list(),
    valid = list(),
    steps = list(),
    iter = list(),
    epoch = NULL,
    epochs = NULL,
    #' @description Initializes an instance of this [R6][R6::R6Class] class.
    initialize = function(epochs, steps_train, steps_valid) {
      self$epochs = assert_count(epochs)

      self$steps$train = steps_train
      self$steps$valid = steps_valid

      self$iter = list(train = 1, valid = 1)
      self$epoch = 0L
    },
    #' @description Appends the value for a measure.
    #' @param measure (`character(1)`)\cr
    #'   The measure that is saved.
    #' @param value (`character(1)`)\cr
    #'   The value for the masure that is being saved.
    #' @param phase (`character(1`)\cr
    #'   The phase, either "train" or "valid".
    append = function(measure, value, phase) {
      iter = self$iter[[phase]]
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
        self$iter$train = 1L
        self$iter$valid = 1L
      } else if (what %in% c("train", "valid")) {
        self$iter[[what]] = self$iter[[what]] + 1L
      } else {
        stopf("Invalid x.")
      }
      invisible(self)
    },
    #' @description Prints the object.
    #' @param ... (any)\cr
    #'   Currently unused.
    print = function(...) {
      catf("<History>")
      catf(" Epoch: %s/%s", self$epoch, self$epochs)
      catf(" Tracked Measures:")
      catf("  * Train: %s", paste(names(self$train), collapse = ","))
      catf("  * Valid: %s", paste(names(self$valid), collapse = ","))
    }
  )
)
