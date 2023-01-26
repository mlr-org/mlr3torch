CallbackTorchHistory = callback_torch(
  public = list(
    initialize = function(ctx) {
      super$initialize(id = "history", label = "Callback Torch History", man = "mlr_callbacks_torch.history")
    }
  ),
  on_begin = function(ctx) {
    self$state$hist_train = list(list(epoch = numeric(0)))
    self$state$hist_valid = list(list(epoch = numeric(0)))
  },
  on_end = function(ctx) {
    self$state$hist_train = rbindlist(self$state$hist_train, fill = TRUE)
    self$state$hist_valid = rbindlist(self$state$hist_valid, fill = TRUE)
  },
  on_before_validation = function(ctx) {
    if (length(ctx$last_scores_train)) {
      self$state$hist_train[[length(self$state$hist_train) + 1]] = c(list(epoch = ctx$epoch), ctx$last_scores_train)
    }
  },
  on_epoch_end = function(ctx) {
    if (length(ctx$last_scores_valid)) {
      self$state$hist_valid[[length(self$state$hist_valid) + 1]] = c(list(epoch = ctx$epoch), ctx$last_scores_valid)
    }
  },
  private = list(
    deep_clone = function(name, value) {
      if (name == "state") {
        map(value, data.table::copy)
      } else {
        value
      }
    }
  )
)
