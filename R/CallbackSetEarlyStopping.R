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
        self$prev_valid_scores = ctx$last_scores_valid
        return(NULL)
      }
      self$prev_valid_scores = ctx$last_scores_valid
      minimize = ctx$measures_valid[[1L]]$minimize

      delta = ctx$last_scores_valid - self$prev_valid_scores
      if (is.na(delta)) {
        lg$warn("Learner %s in epoch %s: Difference between subsequent validation performances is NA",
          ctx$learner$id, ctx$epoch)
        return(NULL)
      }

      delta = if (!minimize) -delta

      if (delta < min_delta) {
        self$stagnation = self$stagnation + 1L
        if (self$stagnation == self$patience) {
          ctx$end_training = TRUE
        }
      } else {
        self$stagnation = 0
      }
    }
  )
)
