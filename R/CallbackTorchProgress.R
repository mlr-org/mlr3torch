#' @title Shows Training Process in the Console
#' @description
#' Prints a progress bar and the metrics for training and validation.
#' @include CallbackTorch.R
#' @export
CallbackTorchProgress = callback_torch(
  public = list(
    initialize = function() {
      super$initialize(id = "progress", label = "Callack Torch Progress", man = "mlr_callbacks_torch.progress")
    }
  ),
  on_epoch_begin = function(ctx) {
    catf("Epoch %s", ctx$epoch)
    self$pb_train = progress::progress_bar$new(
      total = length(ctx$loader_train),
      format = "Training [:bar]"
    )
  },
  on_batch_end = function(ctx) {
    self$pb_train$tick(tokens = list(loss = ctx$last_loss))
  },
  on_before_validation = function(ctx) {
    self$pb_valid = progress::progress_bar$new(
      total = length(ctx$loader_valid),
      format = "Validation: [:bar]"
    )
  },
  on_batch_valid_end = function(ctx) {
    self$pb_valid$tick(tokens = list(loss = last_loss))
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
  }
)

