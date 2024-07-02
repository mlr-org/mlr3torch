#' @title History Callback
#'
#' @name mlr_callback_set.history
#'
#' @description
#' Saves the training and validation history during training.
#' The history is saved as a data.table in the `$train` and `$valid` slots.
#' The first column is always `epoch`.
#'
#' @export
#' @include CallbackSet.R
CallbackSetHistory = R6Class("CallbackSetHistory",
  inherit = CallbackSet,
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
    state_dict = function() {
      list(
        train = rbindlist(self$train, fill = TRUE),
        valid = rbindlist(self$valid, fill = TRUE)
      )
    },
    #' @description
    #' Sets the field `$train` and `$valid` to those contained in the state dict.
    #' @param state_dict (`callback_state_history`)\cr
    #'   The state dict as retrieved via `$state_dict()`.
    load_state_dict = function(state_dict) {
      assert_list(state_dict, "data.table")
      assert_permutation(names(state_dict), c("train", "valid"))
      self$train = state_dict$train
      self$valid = state_dict$valid
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



#' @include TorchCallback.R
mlr3torch_callbacks$add("history", function() {
  TorchCallback$new(
    callback_generator = CallbackSetHistory,
    param_set = ps(),
    id = "history",
    label = "History",
    man = "mlr3torch::mlr_callback_set.history"
  )
})
