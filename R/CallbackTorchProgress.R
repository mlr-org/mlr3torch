#' @title Shows Training Process in the Console
#'
#' @name mlr3torch_callbacks.progress
#'
#' @description
#' Prints a progress bar and the metrics for training and validation.
#'
#' @family callback
#' @include CallbackTorch.R
#' @export
CallbackTorchProgress = R6Class("CallbackTorchProgress",
  inherit = CallbackTorch,
  lock_objects = FALSE,
  private = list(
    on_epoch_begin = function(ctx) {
      catf("Epoch %s", ctx$epoch)
      self$pb_train = progress::progress_bar$new(
        total = length(ctx$loader_train),
        format = "Training [:bar]"
      )
      self$pb_train$tick(0)
    },
    on_batch_end = function(ctx) {
      self$pb_train$tick()
    },
    on_before_valid = function(ctx) {
      self$pb_valid = progress::progress_bar$new(
        total = length(ctx$loader_valid),
        format = "Validation: [:bar]"
      )
      self$pb_valid$tick(0)
    },
    on_batch_valid_end = function(ctx) {
      self$pb_valid$tick()
    },
    on_epoch_end = function(ctx) {
      scores = list()
      scores$train = ctx$last_scores_train
      scores$valid = ctx$last_scores_valid

      scores = Filter(function(x) length(x) > 0, scores)

      if (!length(scores)) {
        catf("[End of epoch %s]", ctx$epoch)
      } else {
        catf("\n[Summary epoch %s]", ctx$epoch)
        cat("------------------\n")
        for (phase in names(scores)) {
          catf("Measures (%s):", capitalize(phase))
          curscore = scores[[phase]]
          output = sprintf(" * %s = %.2f\n", names(curscore), unlist(curscore))
          cat(paste(output, collapse = ""))
        }
      }
    },
    on_end = function(ctx) {
      self$pb_train = NULL
      self$pb_valid = NULL
    }
  )
)

#' @include TorchCallback.R CallbackTorch.R
mlr3torch_callbacks$add("progress", function() {
  TorchCallback$new(
    callback_generator = CallbackTorchProgress,
    param_set = ps(),
    id = "progress",
    label = "Progress",
    man = "mlr3torch::mlr3torch_callbacks.progress",
    packages = "progress"
  )
})
