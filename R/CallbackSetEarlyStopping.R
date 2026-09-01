CallbackSetEarlyStopping = R6Class("CallbackSetEarlyStopping",
  inherit = CallbackSet,
  lock_objects = FALSE,
  public = list(
    # checkpointing runs on_end, while we restore best weights on_exit, so restoring runs
    # without influencing what the checkpoint callback writes to disk
    weight = 1000,
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
        # We need >= so it works with resuming
        if (self$stagnation >= self$patience) {
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
      # far are still the right ones to keep. Callbacks that write the network out run earlier
      # so this does not changes what gets written to disk.
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
        stagnation = self$stagnation,
        measure = self$ctx$measures_valid[[1L]]$id
      )
    },
    load_state_dict = function(state_dict) {
      measure = self$ctx$measures_valid[[1L]]$id
      if (!is.null(state_dict$measure) && !identical(state_dict$measure, measure)) {
        stopf("The checkpoint's best score is a value of the validation measure '%s', but this run tracks '%s'.",
          state_dict$measure, measure)
      }
      self$epoch_at_best_score = state_dict$best_epochs
      self$best_score = state_dict$best_score
      # a checkpoint written before this field existed has none; leaving the value alone keeps it
      # consistent with the `best_epochs` that is being restored alongside it
      if (!is.null(state_dict$best_valid_scores)) {
        self$best_valid_scores = state_dict$best_valid_scores
      }
      self$stagnation = state_dict$stagnation
      if (self$stagnation >= self$patience) {
        warningf("Early stopping had already ended the run this checkpoint belongs to (stagnation %i, patience %i), so this run trains no epoch and returns the model of the checkpoint, even though 'epochs' is greater. A run that early stopping ended is finished; start a new run to train further.", # nolint
          self$stagnation, self$patience)
        self$ctx$terminate = TRUE
      }
      if (self$restore_best_weights) {
        # We don't want to store the best_weights, this is unnecessarily expensive
        warningf("'restore_best_weights' is set, but the weights of the best epoch are not part of a checkpoint. Unless this run improves on the restored best score, it ends with the weights of its last epoch while still reporting the earlier best epoch as its internally tuned 'epochs'.") # nolint
      }
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
