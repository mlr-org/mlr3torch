#' @title Progress Callback
#'
#' @name mlr_callback_set.progress
#'
#' @description
#' Prints a progress bar and the metrics for training and validation.
#'
#' @family Callback
#' @include CallbackSet.R
#' @param digits `integer(1)`\cr
#'   The number of digits to print for the measures.
#' @export
#' @examplesIf torch::torch_is_installed()
#' task = tsk("iris")
#'
#' learner = lrn("classif.mlp", epochs = 5, batch_size = 1,
#'   callbacks = t_clbk("progress"), validate = 0.3)
#' learner$param_set$set_values(
#'   measures_train = msrs(c("classif.acc", "classif.ce")),
#'   measures_valid = msr("classif.ce")
#' )
#'
#' learner$train(task)
CallbackSetProgress = R6Class("CallbackSetProgress",
  inherit = CallbackSet,
  lock_objects = FALSE,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(digits = 2) {
      self$digits = assert_int(digits, lower = 0)
    },
    #' @description
    #' Initializes the progress bar for training.
    on_epoch_begin = function() {
      catf("Epoch %s started (%s)", self$ctx$epoch, format(Sys.time()))
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
      catf("Validation for epoch %s started (%s)", self$ctx$epoch, format(Sys.time()))
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
          output = sprintf(paste0(" * %s = %.", self$digits, "f\n"), names(curscore), unlist(curscore))
          cat(paste(output, collapse = ""))
        }
      }
      cat("\n")
    },
    #' @description
    #' Prints the time at the end of training.
    on_end = function() {
      catf("Finished training for %s epochs (%s)", self$ctx$epoch, format(Sys.time()))
    }
  )
)

#' @include TorchCallback.R
mlr3torch_callbacks$add("progress", function() {
  TorchCallback$new(
    callback_generator = CallbackSetProgress,
    param_set = ps(
      digits = p_int(lower = 1, default = 2, tags = "train")
    ),
    id = "progress",
    label = "Progress",
    man = "mlr3torch::mlr_callback_set.progress",
    packages = "progress"
  )
})
