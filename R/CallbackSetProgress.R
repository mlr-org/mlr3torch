#' @title Progress Callback
#'
#' @name mlr_callback_set.progress
#'
#' @description
#' Prints a progress bar and the metrics for training and validation.
#'
#' @family Callback
#' @include CallbackSet.R
#' @export
CallbackSetProgress = R6Class("CallbackSetProgress",
  inherit = CallbackSet,
  lock_objects = FALSE,
  public = list(
    #' @description
    #' Initializes the progress bar for training.
    on_epoch_begin = function() {
      catf("Epoch %s", self$ctx$epoch)
      self$pb_train = progress::progress_bar$new(
        total = length(self$ctx$loader_train),
        format = "Training [:bar]"
      )
      self$pb_train$tick(0)
    },
    #' @description
    #' Increments the training progress bar.
    on_batch_end = function() {
      self$pb_train$tick()
    },
    #' @description
    #' Creates the progress bar for validation.
    on_before_valid = function() {
      self$pb_valid = progress::progress_bar$new(
        total = length(self$ctx$loader_valid),
        format = "Validation: [:bar]"
      )
      self$pb_valid$tick(0)
    },
    #' @description
    #' Increments the validation progress bar.
    on_batch_valid_end = function() {
      self$pb_valid$tick()
    },
    #' @description
    #' Prints a summary of the training and validation process.
    on_epoch_end = function() {
      scores = list()
      scores$train = self$ctx$last_scores_train
      scores$valid = self$ctx$last_scores_valid

      scores = Filter(function(x) length(x) > 0, scores)

      if (!length(scores)) {
        catf("[End of epoch %s]", self$ctx$epoch)
      } else {
        catf("\n[Summary epoch %s]", self$ctx$epoch)
        cat("------------------\n")
        for (phase in names(scores)) {
          catf("Measures (%s):", capitalize(phase))
          curscore = scores[[phase]]
          output = sprintf(" * %s = %.2f\n", names(curscore), unlist(curscore))
          cat(paste(output, collapse = ""))
        }
      }
    }
  )
)

#' @include TorchCallback.R
mlr3torch_callbacks$add("progress", function() {
  TorchCallback$new(
    callback_generator = CallbackSetProgress,
    param_set = ps(),
    id = "progress",
    label = "Progress",
    man = "mlr3torch::mlr_callback_set.progress",
    packages = "progress"
  )
})
