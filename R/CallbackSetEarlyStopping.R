CallbackSetEarlyStopping = R6Class("CallbackSetEarlyStopping",
  inherit = CallbackSet,
  lock_objects = FALSE,
  public = list(
    # The weight of `Inf` is deliberate and matters in two stages. In `on_valid_end` it makes this
    # callback run last, so it sees the change a user callback made to `ctx$last_scores_valid` --
    # overwriting that field is how a user callback influences early stopping. In `on_exit` it puts
    # the restore of the best weights after `CallbackSetCheckpoint`, which has weight `Inf` as well
    # but is passed by the user and hence comes before this callback, which the learner appends.
    # A checkpoint therefore holds the network as training left it, not the restored one.
    weight = Inf,
    initialize = function(patience, min_delta, restore_best_weights = FALSE) {
      self$patience = assert_int(patience, lower = 1L)
      self$min_delta = assert_double(min_delta, lower = 0, len = 1L, any.missing = FALSE)
      self$restore_best_weights = assert_flag(restore_best_weights)
      self$stagnation = 0L
      self$best_score = NULL
      self$epoch_at_best_score = NULL
      self$best_valid_scores = NULL
      self$best_state_dict = NULL
      self$restored_best_weights = FALSE
    },
    on_valid_end = function() {
      if (is.null(self$ctx$last_scores_valid)) {
        return(NULL)
      }
      if (is.null(self$best_score)) {
        self$best_score = self$ctx$last_scores_valid[[1L]]
        self$epoch_at_best_score = self$ctx$epoch
        self$best_valid_scores = self$ctx$last_scores_valid
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
        self$best_valid_scores = self$ctx$last_scores_valid
        private$.remember_weights()
      }
    },
    on_exit = function() {
      if (!self$restore_best_weights || is.null(self$best_state_dict)) {
        return(NULL)
      }
      # this stage also runs when training was interrupted, in which case the best weights seen so
      # far are still the right ones to keep. Callbacks that write the network out -- such as
      # `CallbackSetCheckpoint` -- have already run at this point, see the `weight` above.
      self$ctx$network$load_state_dict(self$best_state_dict)
      # the learner reads this to decide whether the network it stores is the one of the best epoch,
      # in which case the validation scores of that epoch are the ones describing it
      self$restored_best_weights = TRUE
      invisible(NULL)
    },
    state_dict = function() {
      list(
        # `best_epochs` is what the learner reports as its internally tuned `epochs`
        best_epochs = self$epoch_at_best_score,
        best_score = self$best_score,
        # all validation scores of the epoch at which `best_score` was observed;
        # this is what the learner reports as its `$best_valid_scores`
        best_valid_scores = self$best_valid_scores,
        stagnation = self$stagnation
      )
    },
    load_state_dict = function(state_dict) {
      self$epoch_at_best_score = state_dict$best_epochs
      self$best_score = state_dict$best_score
      self$best_valid_scores = state_dict$best_valid_scores
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
