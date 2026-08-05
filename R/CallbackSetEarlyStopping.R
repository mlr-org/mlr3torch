CallbackSetEarlyStopping = R6Class("CallbackSetEarlyStopping",
  inherit = CallbackSet,
  lock_objects = FALSE,
  public = list(
    # The default weight of 0 is deliberate. This callback is appended after the user's, so in
    # `on_valid_end` it still runs last and therefore sees any change they made to
    # `ctx$last_scores_valid` -- which is how a callback influences early stopping. In `on_exit` a
    # weight of 0 also puts the restore below `CallbackSetCheckpoint` (weight `Inf`), so the
    # checkpoint written there is the restored network rather than the last epoch's.
    initialize = function(patience, min_delta, restore_best_weights = FALSE) {
      self$patience = assert_int(patience, lower = 1L)
      self$min_delta = assert_double(min_delta, lower = 0, len = 1L, any.missing = FALSE)
      self$restore_best_weights = assert_flag(restore_best_weights)
      self$stagnation = 0L
      self$best_score = NULL
      self$epoch_at_best_score = NULL
      self$best_state_dict = NULL
    },
    on_valid_end = function() {
      if (is.null(self$ctx$last_scores_valid)) {
        return(NULL)
      }
      if (is.null(self$best_score)) {
        self$best_score = self$ctx$last_scores_valid[[1L]]
        self$epoch_at_best_score = self$ctx$epoch
        private$.remember_weights()
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
        self$best_score = self$ctx$last_scores_valid[[1L]]
        self$epoch_at_best_score = self$ctx$epoch
        private$.remember_weights()
      }
    },
    on_exit = function() {
      if (!self$restore_best_weights || is.null(self$best_state_dict)) {
        return(NULL)
      }
      # this stage also runs when training was interrupted, in which case the best weights seen so
      # far are still the right ones to keep
      self$ctx$network$load_state_dict(self$best_state_dict)
      invisible(NULL)
    },
    state_dict = function() {
      list(
        # `best_epochs` is what the learner reports as its internally tuned `epochs`
        best_epochs = self$epoch_at_best_score,
        best_score = self$best_score,
        stagnation = self$stagnation
      )
    },
    load_state_dict = function(state_dict) {
      self$epoch_at_best_score = state_dict$best_epochs
      self$best_score = state_dict$best_score
      self$stagnation = state_dict$stagnation
      invisible(NULL)
    }
  ),
  private = list(
    .remember_weights = function() {
      if (!self$restore_best_weights) {
        return(NULL)
      }
      # `$state_dict()` returns the live tensors, so they have to be cloned -- otherwise the
      # "remembered" weights are updated along with the network and restoring them is a no-op.
      # They are kept on the training device, which costs one extra copy of the parameters.
      self$best_state_dict = lapply(self$ctx$network$state_dict(), function(x) x$detach()$clone())
      invisible(NULL)
    }
  )
)
