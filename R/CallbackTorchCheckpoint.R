#' @title Callback Torch Checkpoint
#'
#' @usage NULL
#' @name mlr3torch_callbacks.checkpoint
#' @format `r roxy_format(CallbackTorchCheckpoint)`
#'
#' @description
#' Saves the model during training.
#'
#' @section Construction: `r roxy_construction(CallbackTorchCheckpoint)`
#' * `path` :: `character(1)`\cr
#'   The path to a folder where the models are saved. This path must not exist before.
#' * `freq` :: `integer(1)`\cr
#'   The frequency how often the model is saved. Default is 1.
#' * `save_last` :: `logical(1)`\cr
#'   Whether to always save the last model. Default is `TRUE`.
#'
#' @export
CallbackTorchCheckpoint = R6Class("CallbackTorchCheckpoint",
  inherit = CallbackTorch, lock_objects = FALSE,
  public = list(
    id = "checkpoint",
    initialize = function(path, freq = 1L, save_last = TRUE) {
      assert_path_for_output(path)
      dir.create(path, recursive = TRUE)
      self$path = path
      self$freq = assert_int(freq, lower = 1L)
      self$save_last = assert_flag(save_last)
    },
    on_epoch_end = function(ctx) {
      if ((ctx$epoch %% self$freq) == 0) {
        torch::torch_save(ctx$network, file.path(self$path, paste0("network", ctx$epoch, ".rds")))
      }
    },
    on_end = function(ctx) {
      if (self$save_last) {
        path = file.path(self$path, paste0("network", ctx$epoch, ".rds"))
        if (!file.exists(path)) { # no need to save the last network twice if it was already saved.
          torch::torch_save(ctx$network, path)
        }
      }
    }
  )
)

#' @include CallbackTorch.R
mlr3torch_callbacks$add("checkpoint", function() {
  TorchCallback$new(
    callback_generator = CallbackTorchCheckpoint,
    param_set = ps(
      path = p_uty(),
      freq = p_int(default = 1L, lower = 1L),
      save_last = p_lgl(default = TRUE)
    )
  )
})
