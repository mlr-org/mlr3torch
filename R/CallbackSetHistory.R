#' @title History Callback
#'
#' @name mlr_callback_set.history
#'
#' @description
#' Saves the training and validation history during training.
#' The history is saved as a data.table where the validation measures are prefixed with `"valid."`
#' and the training measures are prefixed with `"train."`.
#'
#' @export
#' @include CallbackSet.R
#' @examplesIf torch::torch_is_installed()
#'
#' cb = t_clbk("history")
#' task = tsk("iris")
#'
#' learner = lrn("classif.mlp", epochs = 3, batch_size = 1,
#'   callbacks = t_clbk("history"), validate = 0.3)
#' learner$param_set$set_values(
#'   measures_train = msrs(c("classif.acc", "classif.ce")),
#'   measures_valid = msr("classif.ce")
#' )
#' learner$train(task)
#'
#' print(learner$model$callbacks$history)
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
      train = rbindlist(self$train, fill = TRUE)
      colnames(train)[-1L] = paste0("train.", colnames(train)[-1L])
      valid = rbindlist(self$valid, fill = TRUE)
      colnames(valid)[-1L] = paste0("valid.", colnames(valid)[-1L])
      state = if (nrow(valid) == 0 && nrow(train) == 0) {
        data.table(epoch = numeric(0))
      } else if (nrow(valid) == 0) {
        train
      } else if (nrow(train) == 0) {
        valid
      } else {
        merge(train, valid, by = "epoch")
      }
      if (is.null(self$prev_state)) {
        state
      } else {
        rbind(state, self$prev_state)
      }
    },
    #' @description
    #' Sets the field `$train` and `$valid` to those contained in the state dict.
    #' @param state_dict (`callback_state_history`)\cr
    #'   The state dict as retrieved via `$state_dict()`.
    load_state_dict = function(state_dict) {
      self$prev_state = state_dict
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
