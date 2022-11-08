#' @title Callback for Torch History
#'
#' @description
#' This predefined callback updates the training and validation history
#'
#' @export
CallbackTorchHistory = callback_torch(
  public = list(
    initialize = function()  {
      super$initialize(id = "history", label = "Callback Torch History", man = "mlr_callbacks_torch.history")
    }
  ),
  on_begin = function(ctx) {
    self$hist_train = list(list(epoch = numeric(0)))
    self$valid_hist = list(list(epoch = numeric(0)))
  },
  on_end = function(ctx) {
    self$hist_train = rbindlist(self$hist_train, fill = TRUE)
    self$valid_hist = rbindlist(self$valid_hist, fill = TRUE)
  },
  on_before_validation = function(ctx) {
    if (length(self$state$last_scores_train)) {
      self$hist_train[[length(self$hist_train) + 1]] = c(list(epoch = self$state$epoch), self$state$last_scores_train)
    }
  },
  on_epoch_end = function(ctx) {
    if (length(self$state$last_scores_valid)) {
      self$hist_valid[[length(self$hist_valid) + 1]] = c(list(epoch = self$state$epoch), self$state$last_scores_valid)
    }
  }
)
