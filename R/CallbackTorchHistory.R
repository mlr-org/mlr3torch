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
  public = list(
    #' @description
    #' Initializes lists where the train and validation metrics are stored.
    #' @param ctx [ContextTorch]
    on_begin = function(ctx) {
      self$train = list(list(epoch = numeric(0)))
      self$valid = list(list(epoch = numeric(0)))
    },
    #' @description
    #' Converts the lists to data.tables.
    #' @param ctx [ContextTorch]
    on_end = function(ctx) {
      self$train = rbindlist(self$train, fill = TRUE)
      self$valid = rbindlist(self$valid, fill = TRUE)
    },
    #' @description
    #' Add the last training scores to the history.
    #' @param ctx [ContextTorch]
    on_before_valid = function(ctx) {
      if (length(ctx$last_scores_train)) {
        self$train[[length(self$train) + 1]] = c(list(epoch = ctx$epoch), ctx$last_scores_train)
      }
    },
    #' @description
    #' Add the last validation scores to the history.
    #' @param ctx [ContextTorch]
    on_epoch_end = function(ctx) {
      if (length(ctx$last_scores_valid)) {
        self$valid[[length(self$valid) + 1]] = c(list(epoch = ctx$epoch), ctx$last_scores_valid)
      }
    }
  ),
  private = list(
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
