#' @title Progress Callback
#'
#' @name mlr_callback_set.progress
#'
#' @description
#' Prints a progress bar and the metrics for training and validation.
#'
#' @section Resuming:
#' This callback can be resumed without any problems.
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
    #' Starts this run's timer.
    on_begin = function() {
      private$.started = Sys.time()
    },
    #' @description
    #' Initializes the progress bar for training.
    on_epoch_begin = function() {
      catf("Epoch %s/%s started (%s)", self$ctx$epoch, self$ctx$total_epochs, format(Sys.time()))
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
    #' Prints the time at the end of training, and how long training took in total.
    #' A resumed run also reports how much of that total it contributed itself.
    on_end = function() {
      total = private$.total_elapsed()
      if (private$.elapsed == 0) {
        catf("Finished training for %s epochs (%s, %.1fs total)", self$ctx$epoch,
          format(Sys.time()), total)
        return(invisible(NULL))
      }
      catf("Finished training for %s epochs (%s, %.1fs total: %.1fs before this run, %.1fs in it)",
        self$ctx$epoch, format(Sys.time()), total, private$.elapsed, total - private$.elapsed)
    },
    #' @description
    #' Returns the seconds trained so far, so that a resumed run reports the time of all runs
    #' together rather than only its own.
    state_dict = function() {
      list(elapsed = private$.total_elapsed())
    },
    #' @description
    #' Loads the time that the previous runs took.
    #' @param state_dict (named `list()`)\cr
    #'   The state dict as retrieved via `$state_dict()`.
    load_state_dict = function(state_dict) {
      private$.elapsed = state_dict$elapsed
      invisible(NULL)
    }
  ),
  private = list(
    # the seconds the runs before this one took, and when this one started
    .elapsed = 0,
    .started = NULL,
    # `$state_dict()` is also called mid-run -- by the checkpoint callback -- so the time this run
    # has taken so far is added on every call rather than only at the end
    .total_elapsed = function() {
      if (is.null(private$.started)) {
        return(private$.elapsed)
      }
      private$.elapsed + as.numeric(difftime(Sys.time(), private$.started, units = "secs"))
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
