#' @title Callback Torch Checkpoint
#'
#' @name mlr_callbacks_torch.checkpoint
#'
#' @description
#' Saves the model during training.
#' @param path (`character(1)`)\cr
#'   The path to a folder where the models are saved. This path must not exist before.
#' @param freq (`integer(1)`)\cr
#'   The frequency how often the model is saved (epoch frequency).
#'
#' @family Callback
#' @export
CallbackTorchCheckpoint = R6Class("CallbackTorchCheckpoint",
  inherit = CallbackTorch,
  lock_objects = FALSE,
  # TODO: This should also save the learner itself
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(path, freq) {
      #   TODO: Maybe we want to be able to give gradient steps here instead of epochs?
      assert_path_for_output(path)
      dir.create(path, recursive = TRUE)
      self$path = path
      self$freq = assert_int(freq, lower = 1L)
    },
    #' @description
    #' Saves the network state dict.
    on_epoch_end = function() {
      if ((self$ctx$epoch %% self$freq) == 0) {
        torch::torch_save(self$ctx$network, file.path(self$path, paste0("network", self$ctx$epoch, ".pt")))
      }
    },
    #' @description
    #' Saves the final network.
    on_end = function() {
      path = file.path(self$path, paste0("network", self$ctx$epoch, ".pt"))
      if (!file.exists(path)) { # no need to save the last network twice if it was already saved.
        torch::torch_save(self$ctx$network, path)
      }
    }
  )
)

#' @include TorchCallback.R CallbackTorch.R
mlr3torch_callbacks$add("checkpoint", function() {
  TorchCallback$new(
    callback_generator = CallbackTorchCheckpoint,
    param_set = ps(
      path      = p_uty(),
      freq      = p_int(lower = 1L)
    ),
    id = "checkpoint",
    label = "Checkpoint",
    man = "mlr3torch::mlr_callbacks_torch.checkpoint"
  )
})
