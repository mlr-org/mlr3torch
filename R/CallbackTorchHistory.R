
#' @export
CallbackTorchHistory = callback_torch(
  on_begin = function() {
    self$hist_train = list(list(epoch = numeric(0)))
    self$valid_hist = list(list(epoch = numeric(0)))
  },
  on_end = function() {
    self$hist_train = rbindlist(self$hist_train, fill = TRUE)
    self$valid_hist = rbindlist(self$valid_hist, fill = TRUE)
  },
  on_before_validation = function() {
    if (length(self$state$last_scores_train)) {
      self$hist_train[[length(self$hist_train) + 1]] = c(list(epoch = self$state$epoch), self$state$last_scores_train)
    }
  },
  on_epoch_end = function() {
    if (length(self$state$last_scores_valid)) {
      self$hist_valid[[length(self$hist_valid) + 1]] = c(list(epoch = self$state$epoch), self$state$last_scores_valid)
    }
  }
)
