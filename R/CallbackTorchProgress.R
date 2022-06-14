#' @title Shows Training Process in the Console
#' @description
#' Prints a progress bar and the metrics for training and validation.
#' @export
#' @include callbacks.R
CallbackTorchProgress = R6Class("CallbackTorchProgress",
  inherit = CallbackTorch,
  lock_objects = FALSE,
  public = list(
    #' @description Initializes an instance of this [R6][R6::R6Class] class.
    initialize = function() {
      super$initialize(id = "progress")
    },
    #' @description
    #' Initializes the progress bar for the training.
    #' @param context (`Context`)\cr
    #'   The context in which the callback is called.
    on_before_train_epoch = function(context) {
      catf("Epoch %s", context$history$epoch)
      self$pb_train = progress_bar$new(
        total = context$history$steps$train,
        format = "Training [:bar]"
      )
    },
    #' @description
    #' Updates the progress bar for the training.
    #' @param context (`Context`)\cr
    #'   The context in which the callback is called.
    on_after_train_batch = function(context) {
      self$pb_train$tick()
    },
    #' @description
    #' Initializes the progress bar for the validation.
    #' @param context (`Context`)\cr
    #'   The context in which the callback is called.
    on_before_valid_epoch = function(context) {
      self$pb_valid = progress_bar$new(
        total = context$history$steps$valid,
        format = "Validation: [:bar]"
      )
    },
    #' @description
    #' Updates the progress bar for the validation.
    #' @param context (`Context`)\cr
    #'   The context in which the callback is called.
    on_after_valid_batch = function(context) {
      self$pb_valid$tick(tokens = list(loss = context$history$last_valid_loss))
    },
    #' @description
    #' Prints the results of the epoch.
    #' @param context (`Context`)\cr
    #'   The context in which the callback is called.
    on_after_valid_epoch = function(context) {
      catf("\n[Summary epoch %s]", context$history$epoch)
      catf("------------------")
      history = context$history
      for (phase in c("train", "valid")) {
        if (length(names(history[[phase]]))) {
          catf("Measures %s:", capitalize(phase))
          for (train_measure in names(history[[phase]])) {
            values = history[[phase]][[train_measure]][[history$epoch]]
            values = unlist(values)
            avg = mean(values)
            catf(" * %s = %.2f", train_measure, avg)
          }
        }
      }
    }
  )
)

torch_callbacks$add("torch.progress", CallbackTorchProgress)
