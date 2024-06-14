#' @title Checkpoint Callback
#'
#' @name mlr_callback_set.checkpoint
#'
#' @description
#' Saves the optimizer and network states during training.
#' The final network and optimizer are always stored.
#' @details
#' Saving the learner itself in the callback with a trained model is impossible,
#' as the model slot is set *after* the last callback step is executed.
#'
#' @param path (`character(1)`)\cr
#'   The path to a folder where the models are saved.
#' @param freq (`integer(1)`)\cr
#'   The frequency how often the model is saved.
#'   Frequency is either per step or epoch, which can be configured through the `freq_type` parameter.
#' @param freq_type (`character(1)`)\cr
#'   Can be be either `"epoch"` (default) or `"step"`.
#' @family Callback
#' @export
#' @include CallbackSet.R
CallbackSetCheckpoint = R6Class("CallbackSetCheckpoint",
  inherit = CallbackSet,
  lock_objects = FALSE,
  # TODO: This should also save the learner itself
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(path, freq, freq_type = "epoch") {
      self$freq = assert_int(freq, lower = 1L)
      self$path = assert_path_for_output(path)
      self$freq_type = assert_choice(freq_type, c("epoch", "step"))
      if (!dir.exists(path)) {
        dir.create(path, recursive = TRUE)
      }
    },
    #' @description
    #' Saves the network and optimizer state dict.
    #' Does nothing if `freq_type` or `freq` are not met.
    on_epoch_end = function() {
      if (self$freq_type == "step" || (self$ctx$epoch %% self$freq != 0)) {
        return(NULL)
      }
      private$.save(self$ctx$epoch)
    },
    #' @description
    #' Saves the selected objects defined in `save`.
    #' Does nothing if freq_type or freq are not met.
    on_batch_end = function() {
      if (self$freq_type == "epoch" || (self$ctx$step %% self$freq != 0)) {
        return(NULL)
      }
      private$.save(self$ctx$step)
    },
    #' @description
    #' Saves the learner.
    on_exit = function() {
      if (self$ctx$epoch == 0) return(NULL)
      if (self$freq_type == "epoch") {
        if (self$ctx$epoch %% self$freq == 0) {
          # already saved
          return(NULL)
        } else {
          private$.save(self$ctx$epoch)
        }
      }
      if (self$freq_type == "step") {
        if (self$ctx$step %% self$freq == 0) {
          # already saved
          return(NULL)
        } else {
          private$.save(self$ctx$epoch)
        }
      }
    }
  ),
  private = list(
    .save = function(suffix) {
      torch_save(self$ctx$network$state_dict(), file.path(self$path, paste0("network", suffix, ".pt")))
      torch_save(self$ctx$optimizer$state_dict(), file.path(self$path, paste0("optimizer", suffix, ".pt")))
    }
  )
)

#' @include TorchCallback.R
mlr3torch_callbacks$add("checkpoint", function() {
  TorchCallback$new(
    callback_generator = CallbackSetCheckpoint,
    param_set = ps(
      path      = p_uty(tags = c("train", "required")),
      freq      = p_int(lower = 1L, tags = c("train", "required")),
      freq_type = p_fct(default = "epoch", c("epoch", "step"), tags = "train")
    ),
    id = "checkpoint",
    label = "Checkpoint",
    man = "mlr3torch::mlr_callback_set.checkpoint"
  )
})
