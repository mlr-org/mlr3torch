CallbackSetEarlyStopping = R6Class("CallbackSetEarlyStopping",
  inherit = CallbackSet,
  lock_objects = FALSE,
  public = list(
    initialize = function(patience, min_delta) {
      self$patience = assert_int(patience, lower = 1L)
      self$min_delta = assert_double(min_delta, lower = 0, len = 1L, any.missing = FALSE)
      self$stagnation = 0L
      self$best_score = NULL
    },
    on_valid_end = function() {
      if (is.null(self$ctx$last_scores_valid)) {
        return(NULL)
      }
      if (is.null(self$best_score)) {
        self$best_score = self$ctx$last_scores_valid[[1L]]
        return(NULL)
      }
      multiplier = if (self$ctx$measures_valid[[1L]]$minimize) -1 else 1
      improvement = multiplier * (self$ctx$last_scores_valid[[1L]] - self$best_score)

      if (is.na(improvement)) {
        lg$warn("Learner %s in epoch %s: Difference between subsequent validation performances is NA",
          self$ctx$learner$id, self$ctx$epoch)
        return(NULL)
      }

      if (improvement <= self$min_delta) {
        self$stagnation = self$stagnation + 1L
        if (self$stagnation == self$patience) {
          self$ctx$terminate = TRUE
        }
      } else {
        self$stagnation = 0
      }
      if (improvement > 0) {
        self$best_score = self$ctx$last_scores_valid[[1L]]
      }
    }
  )
)
