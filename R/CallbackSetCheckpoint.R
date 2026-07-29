#' @title Checkpoint Callback
#'
#' @name mlr_callback_set.checkpoint
#'
#' @description
#' Saves the optimizer and network states during training.
#' The final network and optimizer are always stored.
#'
#' For a checkpoint written after `<n>` epochs (or steps, see `freq_type`), three files are created
#' in `path`:
#' * `network<n>.pt` :: The `$state_dict()` of the network.
#' * `optimizer<n>.pt` :: The `$state_dict()` of the optimizer.
#' * `state<n>.rds` :: The epoch and step the checkpoint was written in, as well as the
#'   `$state_dict()`s of the other callbacks.
#'
#' Training can be continued from such a checkpoint -- also in a new R session -- using
#' [`CallbackSetResume`], see the example below.
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
#' 
#' @examplesIf torch::torch_is_installed()
#' cb = t_clbk("checkpoint", freq = 1)
#' task = tsk("iris")
#' 
#' pth = tempfile()
#' learner = lrn("classif.mlp", epochs = 3, batch_size = 1, callbacks = cb)
#' learner$param_set$set_values(cb.checkpoint.path = pth)
#' 
#' learner$train(task)
#'
#' list.files(pth)
#'
#' # continue training for 3 more epochs, starting from the last checkpoint
#' learner_resumed = lrn("classif.mlp", epochs = 6, batch_size = 1,
#'   callbacks = t_clbk("resume"))
#' learner_resumed$param_set$set_values(cb.resume.path = pth)
#' learner_resumed$train(task)
#' learner_resumed$model$epochs
CallbackSetCheckpoint = R6Class("CallbackSetCheckpoint",
  inherit = CallbackSet,
  lock_objects = FALSE,
  # TODO: This should also save the learner itself
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(path, freq, freq_type = "epoch") {
      self$freq = assert_int(freq, lower = 1L)
      self$freq_type = assert_choice(freq_type, c("epoch", "step"))
      # a folder that already contains checkpoints may be continued in, which is what happens when
      # a run is resumed via CallbackSetResume. Any other existing path is rejected, so that
      # unrelated data is never written into.
      self$path = if (is_checkpoint_dir(path)) path else assert_path_for_output(path)
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
          private$.save(self$ctx$step)
        }
      }
    }
  ),
  private = list(
    .save = function(suffix) {
      torch_save(self$ctx$network$state_dict(), file.path(self$path, paste0("network", suffix, ".pt")))
      torch_save(self$ctx$optimizer$state_dict(), file.path(self$path, paste0("optimizer", suffix, ".pt")))
      # the training state is what [`CallbackSetResume`] needs on top of the network and optimizer:
      # the epoch to continue from and the states of the other callbacks.
      # Callback states are plain R objects -- they are not torch-serialized when a learner is
      # marshaled either -- so they are saved with saveRDS(), which (unlike torch_save()) keeps
      # classes such as data.table intact.
      saveRDS(
        list(
          epoch     = self$ctx$epoch,
          step      = self$ctx$step,
          callbacks = discard(map(self$ctx$callbacks, function(cb) cb$state_dict()), is.null)
        ),
        file.path(self$path, paste0("state", suffix, ".rds"))
      )
    }
  )
)

# Whether `path` looks like a folder that CallbackSetCheckpoint has written to.
# Used to decide whether it is safe to write more checkpoints into an existing folder.
is_checkpoint_dir = function(path) {
  dir.exists(path) &&
    length(list.files(path, pattern = "^(network|optimizer)[0-9]+\\.pt$|^state[0-9]+\\.rds$")) > 0L
}

# The suffixes of the complete checkpoints in `path`, most recent first.
checkpoint_suffixes = function(path) {
  if (!dir.exists(path)) return(integer(0))
  suffixes = as.integer(gsub("^network|\\.pt$", "", list.files(path, pattern = "^network[0-9]+\\.pt$")))
  # a checkpoint is only usable if the matching optimizer was written as well, which is not the
  # case when a run was interrupted between the two
  sort(suffixes[file.exists(file.path(path, paste0("optimizer", suffixes, ".pt")))], decreasing = TRUE)
}

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
