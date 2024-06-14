CallbackSetEarlyStopping = R6Class("CallbackSetEarlyStopping",
  inherit = CallbackSet,
  lock_objects = FALSE,
  public = list(
    initialize = function(patience, min_delta) {
      self$patience = assert_int(patience, lower = 1L)
      self$min_delta = assert_double(min_delta, len = 1L, any.missing = FALSE)
      self$stagnation = 0L
    },
    on_valid_end = function() {
      if (is.null(self$prev_valid_scores)) {
        self$prev_valid_scores = self$ctx$last_scores_valid
        return(NULL)
      }
      if (is.null(self$ctx$last_scores_valid)) {
        return(NULL)
      }
      self$prev_valid_scores = self$ctx$last_scores_valid
      minimize = self$ctx$measures_valid[[1L]]$minimize

      delta = self$ctx$last_scores_valid[[1L]] - self$prev_valid_scores[[1L]]
      if (is.na(delta)) {
        lg$warn("Learner %s in epoch %s: Difference between subsequent validation performances is NA",
          self$ctx$learner$id, self$ctx$epoch)
        return(NULL)
      }

      delta = if (!minimize) -delta else delta

      if (delta < self$min_delta) {
        self$stagnation = self$stagnation + 1L
        if (self$stagnation == self$patience) {
          self$ctx$end_training = TRUE
        }
      } else {
        self$stagnation = 0
      }
    }
  )
)
