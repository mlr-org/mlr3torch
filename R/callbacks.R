#' @export
torch_callbacks = mlr3misc::Dictionary$new()

#' @title Retrieve a callback
#' @export
cllb = function(id, ...) {
  torch_callbacks$get(id, ...)
}

#' @title Retrieve callbacks
#' @export
cllbs = function(ids, ...) {
  map(ids, function(id) cllb(id, ...))

}


#' @title Callbacks for Torch Learner
#' @description
#' All Callbacks for the Torch Learners should inherit from this class.
#' @export
CallbackTorch = R6Class("CallbackTorch",
  inherit = Callback,
  public = list(
    #' @description Initializes an instance of this [R6][R6::R6Class] class.
    #' @param id (`character(1)`)\cr
    #'   The id for the new object.
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
    },
    #' @field context ([`ContextTorch`][ContextTorch])\cr
    #'   The context in which'the callbacks are evaluated.
    context = NULL
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
    #' @description Initializes an instance of this [R6][R6::R6Class] class.
    initialize = function() {
      super$initialize(id = "progress")
    },
    #' @description
    #' Initializes the progress bar for the training.
    on_before_train_epoch = function() {
      catf("Epoch %s", self$context$history$epoch)
      self$pb_train = progress::progress_bar$new(
        total = self$context$history$steps$train,
        format = "Training [:bar]"
      )
    },
    #' @description
    #' Updates the progress bar for the training.
    on_after_train_batch = function() {
      self$pb_train$tick()
    },
    #' @description
    #' Initializes the progress bar for the validation.
    on_before_valid_epoch = function() {
      self$pb_valid = progress::progress_bar$new(
        total = self$context$history$steps$valid,
        format = "Validation: [:bar]"
      )
    },
    #' @description
    #' Updates the progress bar for the validation.
    on_after_valid_batch = function() {
      self$pb_valid$tick(tokens = list(loss = self$context$history$last_train_loss))
    },
    #' @description
    #' Prints the results of the epoch.
    on_after_valid_epoch = function() {
      catf("\n[Summary epoch %s]", self$context$history$epoch)
      catf("------------------")
      history = self$context$history
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
