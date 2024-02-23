#' @title Checkpoint Callback
#'
#' @name mlr_callback_set.checkpoint
#'
#' @description
#' Saves the optimizer and network states during training.
#'
#' @param path (`character(1)`)\cr
#'   The path to a folder where the models are saved. This path must not exist before.
#' @param freq (`integer(1)`)\cr
#'   The frequency how often the model is saved (epoch frequency).
#'   Frequency is either per step or epoch, which can be configured through the `freq_type` parameter.
#' @param save (`character()`)\cr
#'   The objects to save, must be either "all" or a subset of "network", "optimizer", "learner" and "loss".
#'   Default is "all".
#'   Only the network and optimizer are stored every `freq` `epoch`s or `step`s, as they are necesseary to resume
#'   training for a specific point.
#'   The (final) learner and the loss are only stored once at the end of training.
#' @param freq_type (`character(1)`)\cr
#'   Can be be either `"epoch"` (default) or `"step"`.
#'
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
    initialize = function(path, freq = 1, save = "all", freq_type = "epoch") {
      self$freq = assert_int(freq, lower = 1L)
      if (isTRUE(all.equal(save, "all"))) {
        self$save = c("network", "optimizer", "learner", "loss")
      } else {
        self$save = assert_subset(save, c("network", "optimizer", "learner", "loss"))
      }
      self$freq_type = assert_subset(freq_type, c("epoch", "step"))
      assert(check_directory_exists(path), check_path_for_output(path))
      self$path = path
      if (!dir.exists(path)) {
        dir.create(path, recursive = TRUE)
      }
    },
    #' @description
    #' Saves the configuration to a file.
    on_begin = function() {
      jsonlite::write_json(list(
        freq = self$freq,
        freq_type = self$freq_type,
        save = self$save
      ), path = file.path(self$path, "config.json"))
    },
    #' @description
    #' Saves the objects network and optimizer if selected.
    #' Does nothing if freq_type or freq are not met.
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
    #' Saves the objects network and optimizer and learner if selected.
    #' If the learner is already saved it will not be saved again.
    on_exit = function() {
      path_final = file.path(self$path, "final")
      if ("learner" %in% self$save) {
        # FIXME: marshal first once implemented
        saveRDS(self$ctx$learner, file.path(self$path, "learner.rds"))
      }
      if ("loss" %in% self$save) {
        # e.g. cross_entropy loss can have weights
        torch::torch_save(self$ctx$loss_fn$state_dict(), file.path(self$path, "loss.pt"))
      }
    }
  ),
  private = list(
    .save = function(suffix) {
      if ("network" %in% self$save) {
        torch_save(self$ctx$network$state_dict(), file.path(self$path, paste0("network", suffix, ".pt")))
      }
      if ("optimizer" %in% self$save) {
        torch_save(self$ctx$optimizer$state_dict(), file.path(self$path, paste0("optimizer", suffix, ".pt")))
      }
    }
  )
)

#' @include TorchCallback.R
mlr3torch_callbacks$add("checkpoint", function() {
  TorchCallback$new(
    callback_generator = CallbackSetCheckpoint,
    param_set = ps(
      path =      p_uty(tags = c("train", "required")),
      freq =      p_int(default = 1L, lower = 1L, tags = "train"),
      freq_type = p_fct(default = "epoch", levels = c("epoch", "step"), tags = "train"),
      save =      p_uty(default = "all", tags = "train", custom_check = crate(function(x) {
        if (isTRUE(all.equal(x, "all"))) return(TRUE)
        check_subset(x, c("network", "optimizer", "learner", "loss"), empty.ok = FALSE)}, .parent = topenv()))
    ),
    id = "checkpoint",
    label = "Checkpoint",
    packages = "jsonlite",
    man = "mlr3torch::mlr_callback_set.checkpoint"
  )
})
