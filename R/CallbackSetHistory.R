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
    },
    #' @description Plots the history.
    #' @param measures (`character()`)\cr
    #'   Which measures to plot. No default.
    #' @param set (`character(1)`)\cr
    #'   Which set to plot. Either `"train"` or `"valid"`. Default is `"valid"`.
    #' @param epochs (`integer()`)\cr
    #'   An integer vector restricting which epochs to plot. Default is `NULL`, which plots all epochs.
    #' @param theme ([ggplot2::theme()])\cr
    #'   The theme, [ggplot2::theme_minimal()] is the default.
    #' @param ... (any)\cr
    #'   Currently unused.
    plot = function(measures, set = "valid", epochs = NULL, theme = ggplot2::theme_minimal(), ...) {
      assert_choice(set, c("valid", "train"))
      data = self[[set]]
      assert_subset(measures, colnames(data))

      if (is.null(epochs)) {
        data = data[, c("epoch", measures), with = FALSE]
      } else {
        assert_integerish(epochs, unique = TRUE)
        data = data[get("epoch") %in% epochs, c("epoch", measures), with = FALSE]
      }

      if ((!nrow(data)) || (ncol(data) < 2)) {
        stopf("No eligible measures to plot for set '%s'.", set)
      }

      epoch = score = measure = .data = NULL
      if (ncol(data) == 2L) {
        ggplot2::ggplot(data = data, ggplot2::aes(x = epoch, y = .data[[measures]])) +
          ggplot2::geom_line() +
          ggplot2::geom_point() +
          ggplot2::labs(
            x = "Epoch",
            y = measures,
            title = sprintf("%s Loss", switch(set, valid = "Validation", train = "Training"))
          ) +
          theme
      } else {
        data = melt(data, id.vars = "epoch", variable.name = "measure", value.name = "score")
        ggplot2::ggplot(data = data, ggplot2::aes(x = epoch, y = score, color = measure)) +
          viridis::scale_color_viridis(discrete = TRUE) +
          ggplot2::geom_line() +
          ggplot2::geom_point() +
          ggplot2::labs(
            x = "Epoch",
            y = "Score",
            title = sprintf("%s Loss", switch(set, valid = "Validation", train = "Training"))
          ) +
          theme
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
