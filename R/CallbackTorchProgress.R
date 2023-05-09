#' @title Shows Training Process in the Console
#'
#' @name mlr_callbacks_torch.progress
#'
#' @description
#' Prints a progress bar and the metrics for training and validation.
#'
#' @family Callback
#' @include CallbackTorch.R
#' @export
CallbackTorchProgress = R6Class("CallbackTorchProgress",
  inherit = CallbackTorch,
  lock_objects = FALSE,
  public = list(
    #' @description
    #' Initializes the progress bar for training.
    #' @param ctx [ContextTorch]
    on_epoch_begin = function(ctx) {
      catf("Epoch %s", ctx$epoch)
      self$pb_train = progress::progress_bar$new(
        total = length(ctx$loader_train),
        format = "Training [:bar]"
      )
      self$pb_train$tick(0)
    },
    #' @description
    #' Increments the training progress bar.
    #' @param ctx [ContextTorch]
    on_batch_end = function(ctx) {
      self$pb_train$tick()
    },
    #' @description
    #' Creates the progress bar for validation.
    #' @param ctx [ContextTorch]
    on_before_valid = function(ctx) {
      self$pb_valid = progress::progress_bar$new(
        total = length(ctx$loader_valid),
        format = "Validation: [:bar]"
      )
      self$pb_valid$tick(0)
    },
    #' @description
    #' Increments the validation progress bar.
    #' @param ctx [ContextTorch]
    on_batch_valid_end = function(ctx) {
      self$pb_valid$tick()
    },
    #' @description
    #' Prints a summary of the training and validation process.
    #' @param ctx [ContextTorch]
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
    #' @description
    #' Deletes the progess bar objects.
    #' @param ctx [ContextTorch]
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
    man = "mlr3torch::mlr_callbacks_torch.progress",
    packages = "progress"
  )
})
