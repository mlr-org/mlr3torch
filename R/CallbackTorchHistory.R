#' @title Callback Torch History
#'
#' @name mlr3torch_callbacks.history
#'
#' @description
#' Saves the history during training.
#'
#' @export
CallbackTorchHistory = R6Class("CallbackTorchHistory",
  inherit = CallbackTorch,
  lock_objects = FALSE,
  private = list(
    on_begin = function(ctx) {
      self$train = list(list(epoch = numeric(0)))
      self$valid = list(list(epoch = numeric(0)))
    },
    on_end = function(ctx) {
      self$train = rbindlist(self$train, fill = TRUE)
      self$valid = rbindlist(self$valid, fill = TRUE)
    },
    on_before_valid = function(ctx) {
      if (length(ctx$last_scores_train)) {
        self$train[[length(self$train) + 1]] = c(list(epoch = ctx$epoch), ctx$last_scores_train)
      }
    },
    on_epoch_end = function(ctx) {
      if (length(ctx$last_scores_valid)) {
        self$valid[[length(self$valid) + 1]] = c(list(epoch = ctx$epoch), ctx$last_scores_valid)
      }
    },
    deep_clone = function(name, value) {
      if (name %in% c("train", "valid")) {
        data.table::copy(value)
      } else {
        value
      }
    }
  )
)



#' @include TorchCallback.R CallbackTorch.R
mlr3torch_callbacks$add("history", function() {
  TorchCallback$new(
    callback_generator = CallbackTorchHistory,
    param_set = ps(),
    id = "history",
    label = "History",
    man = "mlr3torch::mlr3torch_callbacks.history"
  )
})
