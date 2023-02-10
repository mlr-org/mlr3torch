#' @title Callback Torch History
#'
#' @usage NULL
#' @name mlr3torch_callbacks.history
#' @format `r roxy_format(CallbackTorchHistory)`
#'
#' @description
#' Saves the history during training.
#'
#' @export
CallbackTorchHistory = R6Class("CallbackTorchHistory",
  inherit = CallbackTorch, lock_objects = FALSE,
  public = list(
    id = "history",
    man = "mlr_callbacks_torch.history",
    label = "Torch History",
    on_begin = function(ctx) {
      self$train = list(list(epoch = numeric(0)))
      self$valid = list(list(epoch = numeric(0)))
    },
    on_end = function(ctx) {
      self$train = rbindlist(self$train, fill = TRUE)
      self$valid = rbindlist(self$valid, fill = TRUE)
    },
    on_before_validation = function(ctx) {
      if (length(ctx$last_scores_train)) {
        self$train[[length(self$train) + 1]] = c(list(epoch = ctx$epoch), ctx$last_scores_train)
      }
    },
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



#' @include CallbackTorch.R
mlr3torch_callbacks$add("history", function() {
  TorchCallback$new(
    callback_generator = CallbackTorchHistory,
    param_set = ps()
  )
})

