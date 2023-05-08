#' @title Callback Torch Checkpoint
#'
#' @name mlr3torch_callbacks.checkpoint
#'
#' @description
#' Saves the model during training.
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
    #' @param path (`character(1)`)\cr
    #'   The path to a folder where the models are saved. This path must not exist before.
    #' @param freq (`integer(1)`)\cr
    #'   The frequency how often the model is saved (epoch frequency).
    #' @param save_last (`logical(1)`)\cr
    #'   Whether to always save the last model.
    initialize = function(path, freq, save_last = TRUE) {
      #   TODO: Maybe we want to be able to give gradient steps here instead of epochs?
      assert_path_for_output(path)
      dir.create(path, recursive = TRUE)
      self$path = path
      self$freq = assert_int(freq, lower = 1L)
      self$save_last = assert_flag(save_last)
    },
    #' @description
    #' Saves the network state dict.
    #' @param ctx [ContextTorch]
    on_epoch_end = function(ctx) {
      if ((ctx$epoch %% self$freq) == 0) {
        torch::torch_save(ctx$network, file.path(self$path, paste0("network", ctx$epoch, ".pt")))
      }
    },
    #' @description
    #' Saves Saves the final network.
    #' @param ctx [ContextTorch]
    on_end = function(ctx) {
      if (self$save_last) {
        path = file.path(self$path, paste0("network", ctx$epoch, ".pt"))
        if (!file.exists(path)) { # no need to save the last network twice if it was already saved.
          torch::torch_save(ctx$network, path)
        }
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
      freq      = p_int(lower = 1L),
      save_last = p_lgl(default = TRUE)
    ),
    id = "checkpoint",
    label = "Checkpoint",
    man = "mlr3torch::mlr3torch_callbacks.checkpoint"
  )
})
