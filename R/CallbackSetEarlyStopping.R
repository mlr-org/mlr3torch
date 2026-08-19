CallbackSetEarlyStopping = R6Class("CallbackSetEarlyStopping",
  inherit = CallbackSet,
  lock_objects = FALSE,
  public = list(
    # A high weight, so that `on_valid_end` runs after the user callbacks and this callback sees
    # the change one of them made to `ctx$last_scores_valid` -- overwriting that field is how a
    # user callback influences early stopping. It is deliberately finite, unlike the `Inf` of
    # `CallbackSetCheckpoint`, which has to be last of all: a callback can still be placed between
    # the two.
    # That the restore of the best weights comes after the checkpoint has written is not the
    # weight's doing but the stage's: that callback writes in `on_epoch_end` / `on_end`, and both
    # run before any `on_exit`. A checkpoint therefore holds the network as training left it, not
    # the restored one.
    weight = 1000,
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
        # `>=` rather than `==`: a resumed run restores `stagnation` and can therefore start at or
        # above `patience`, which an equality test would step over and never match again
        if (self$stagnation >= self$patience) {
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
      # far are still the right ones to keep. Callbacks that write the network out -- such as
      # `CallbackSetCheckpoint` -- do so in earlier stages and have already run, see above.
      self$ctx$network$load_state_dict(self$best_state_dict)
      invisible(NULL)
    },
    state_dict = function() {
      # `best_state_dict` is deliberately not part of this: it is a full copy of the network, which
      # every checkpoint would then carry. What that costs a resumed run is warned about below.
      list(
        # `best_epochs` is what the learner reports as its internally tuned `epochs`
        best_epochs = self$epoch_at_best_score,
        best_score = self$best_score,
        stagnation = self$stagnation,
        # which measure `best_score` is a value of, so that a run continuing it can check that it
        # tracks the same one, see $load_state_dict()
        measure = self$ctx$measures_valid[[1L]]$id
      )
    },
    load_state_dict = function(state_dict) {
      # Only the first validation measure is tracked, and `best_score` is a value of it: a run that
      # tracks another one would compare two different scales -- with the new measure's `minimize`
      # direction -- and stop on that difference rather than on a lack of improvement.
      # A checkpoint written before this was recorded carries no measure and is not checked.
      measure = self$ctx$measures_valid[[1L]]$id
      if (!is.null(state_dict$measure) && !identical(state_dict$measure, measure)) {
        stopf("The checkpoint's best score is a value of the validation measure '%s', but this run tracks '%s'. Early stopping compares the two, so set 'measures_valid' as the run that wrote the checkpoint had it -- its first measure is the one early stopping uses.", # nolint
          state_dict$measure, measure)
      }
      self$epoch_at_best_score = state_dict$best_epochs
      self$best_score = state_dict$best_score
      self$stagnation = state_dict$stagnation
      if (self$restore_best_weights) {
        # `best_state_dict` stays NULL until this run improves on the restored score, and $on_exit()
        # then restores nothing. Without this the run would report an epoch it does not hold.
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
