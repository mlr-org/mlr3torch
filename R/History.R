#' @export
History = R6Class("History",
  public = list(
    train_loss = list(),
    valid_loss = list(),
    n_train = NA_integer_,
    n_valid = NA_integer_,
    epoch = NULL,
    train_iter = NULL,
    valid_iter = NULL,
    last_train_loss = NULL,
    initialize = function() {
      self$epoch = 1L
      self$train_iter = 1L
      self$valid_iter = 1L
    },
    add_train_loss = function(loss) {
      self$last_train_loss = loss
      if (length(self$train_loss) < self$epoch) {
        self$train_loss[[self$epoch]] = rep(NA_real_, times = self$n_train)
      }
      self$train_loss[[self$epoch]][[self$train_iter]] = loss
      invisible(self)
    },
    add_valid_loss = function(loss) {
      if (length(self$valid_loss) < self$epoch) {
        self$valid_loss[[self$epoch]] = rep(NA_real_, times = self$n_train)
      }
      self$valid_loss[[self$epoch]][[self$valid_iter]] = loss
      invisible(self)
    }
  )
)
