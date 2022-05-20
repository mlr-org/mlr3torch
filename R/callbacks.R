#' @export
torch_callbacks = mlr3misc::Dictionary$new()

#' @title Retrieve a callback
#' @export
cllb = function(id) {
  torch_callbacks$get(id)
}


#' @title Callbacks for Torch Learner
#' @description
#' All Callbacks for the Torch Learners should inherit from this class.
#' @export
CallbackTorch = R6Class("CallbackTorch",
  inherit = bbotk::Callback,
  public = list(
    initialize = function(id) {
      x = names(self)
      callbacks = x[startsWith(x, "on_")]
      steps = gsub("on_", "", callbacks)
      invalid_steps = steps %nin% private$.steps
      if (sum(invalid_steps > 0L)) {
        message = paste0(
          "Invalid public method(s) ", paste(callbacks[invalid_steps], collapse = ", "), "."
        )
        stopf(message)
      }
      super$initialize(id)
    }
  ),
  private = list(
    .steps = c(
      "start",
      "before_train_epoch",
      "before_train_batch",
      "after_train_batch",
      "before_valid_epoch",
      "before_valid_batch",
      "after_valid_batch",
      "after_valid_epoch",
      "end"
    )
  )
)

#' @title Shows Training Process in the Console
#' @description
#' Prints a progress-bar and the metrics for training and validation.
#' @export
CallbackTorchProgress = R6Class("CallbackTorchProgress",
  inherit = CallbackTorch,
  lock_objects = FALSE,
  public = list(
    initialize = function(id = "progress") {
      super$initialize(id = id)
    },
    on_before_train_epoch = function(context) {
      catf("Epoch %s", context$history$epoch)
      self$pb_train = progress::progress_bar$new(
        total = context$history$steps$train,
        format = "Training [:bar]"
      )
    },
    on_after_train_batch = function(context) {
      self$pb_train$tick()
    },
    on_before_valid_epoch = function(context) {
      self$pb_valid = progress::progress_bar$new(
        total = context$history$steps$valid,
        format = "Validation: [:bar]"
      )
    },
    on_after_valid_batch = function(context) {
      self$pb_valid$tick(tokens = list(loss = context$history$last_train_loss))
    },
    on_after_valid_epoch = function(context) {
      catf("\n[Summary epoch %s]", context$history$epoch)
      catf("----------------")
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
