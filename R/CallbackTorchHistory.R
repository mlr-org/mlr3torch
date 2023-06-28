#' @title Callback Torch History
#'
#' @name mlr_callbacks_torch.history
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
    on_begin = function() {
      self$train = list(list(epoch = numeric(0)))
      self$valid = list(list(epoch = numeric(0)))
    },
    #' @description
    #' Converts the lists to data.tables.
    on_end = function() {
      self$train = rbindlist(self$train, fill = TRUE)
      self$valid = rbindlist(self$valid, fill = TRUE)
    },
    #' @description
    #' Add the latest training scores to the history.
    on_before_valid = function() {
      if (length(self$ctx$last_scores_train)) {
        self$train[[length(self$train) + 1]] = c(
          list(epoch = self$ctx$epoch), self$ctx$last_scores_train
        )
      }
    },
    #' @description
    #' Add the latest validation scores to the history.
    on_epoch_end = function() {
      if (length(self$ctx$last_scores_valid)) {
        self$valid[[length(self$valid) + 1]] = c(
          list(epoch = self$ctx$epoch), self$ctx$last_scores_valid
        )
      }
    }
  ),
  private = list(
    deep_clone = function(name, value) {
      if (name %in% c("train", "valid")) {
        data.table::copy(value)
      } else {
        super$deep_clone(name, value)
      }
    }
  )
)



#' @include DescriptorTorchCallback.R CallbackTorchHistory.R
mlr3torch_callbacks$add("history", function() {
  DescriptorTorchCallback$new(
    callback_generator = CallbackTorchHistory,
    param_set = ps(),
    id = "history",
    label = "History",
    man = "mlr3torch::mlr_callbacks_torch.history"
  )
})
