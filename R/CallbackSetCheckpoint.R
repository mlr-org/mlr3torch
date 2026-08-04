#' @title Checkpoint Callback
#'
#' @name mlr_callback_set.checkpoint
#'
#' @description
#' Saves the optimizer and network states during training.
#' The final network and optimizer are always stored.
#'
#' Checkpoints are written at the end of an epoch. For one written after epoch `<n>`, three files
#' are created in `path`:
#' * `network<n>.pt` :: The `$state_dict()` of the network.
#' * `optimizer<n>.pt` :: The `$state_dict()` of the optimizer.
#' * `state<n>.rds` :: The epoch, as well as the `$state_dict()`s of the other callbacks of the
#'   training run, so that a later run can continue e.g. the training history or the learning rate
#'   schedule.
#'
#' An epoch that was interrupted -- because training failed or was stopped -- is not written under
#' its own number, so `network<n>.pt` is always the network at the *end* of epoch `n`.
#' The network and optimizer of the last *complete* epoch are still written when training is
#' interrupted, but without a `state<n>.rds`: the other callbacks have moved on into the epoch that
#' was interrupted, so their states no longer describe the checkpoint.
#'
#' A folder that already contains checkpoints is accepted, so that a run which continues an earlier
#' one can keep checkpointing into it.
#'
#' Training can be continued from such a checkpoint -- also in a new R session -- via the `path`
#' parameter of [`LearnerTorch`], see the example below.
#' @details
#' Saving the learner itself in the callback with a trained model is impossible,
#' as the model slot is set *after* the last callback step is executed.
#'
#' @param path (`character(1)`)\cr
#'   The path to a folder where the models are saved.
#' @param freq (`integer(1)`)\cr
#'   How often the model is saved, in epochs.
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
#' learner_resumed = lrn("classif.mlp", epochs = 6, batch_size = 1, path = pth)
#' learner_resumed$train(task)
#' learner_resumed$model$epochs
CallbackSetCheckpoint = R6Class("CallbackSetCheckpoint",
  inherit = CallbackSet,
  lock_objects = FALSE,
  # TODO: This should also save the learner itself
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(path, freq) {
      self$freq = assert_int(freq, lower = 1L)
      # an empty folder is a legal target -- that is what a pre-created output directory looks like,
      # and what a run that died before writing its first checkpoint leaves behind -- and so is one
      # that already holds checkpoints, which is what a continued run writes into. Any other
      # existing path is rejected, so that unrelated data is never written into.
      self$path = if (can_checkpoint_into(path)) path else assert_path_for_output(path)
      if (!dir.exists(path)) {
        dir.create(path, recursive = TRUE)
      }
    },
    #' @description
    #' Saves the network and optimizer state dict.
    #' Does nothing if `freq` is not met.
    on_epoch_end = function() {
      # tracked even when nothing is saved here, so that the 'exit' stage knows how many epochs
      # were actually completed
      private$.complete_epochs = self$ctx$epoch
      if (self$ctx$epoch %% self$freq != 0) {
        return(NULL)
      }
      private$.save(self$ctx$epoch)
    },
    #' @description
    #' Saves the final network and optimizer, unless the last complete epoch was already saved.
    on_exit = function() {
      # this stage also runs when training was interrupted, in which case the epoch in progress is
      # deliberately not saved: `network<n>.pt` is meant to be the network at the *end* of epoch n,
      # and the weights of a half-trained epoch are not that.
      if (private$.complete_epochs == 0L || private$.complete_epochs %% self$freq == 0) {
        # nothing was completed, or the last complete epoch was already saved
        return(NULL)
      }
      # the other callbacks have already moved into the epoch that was interrupted, so their
      # states describe neither this checkpoint nor any other one and are not written
      private$.save(private$.complete_epochs, save_state = private$.complete_epochs == self$ctx$epoch)
    }
  ),
  private = list(
    # the number of epochs that were trained to completion
    .complete_epochs = 0L,
    .save = function(suffix, save_state = TRUE) {
      torch_save(self$ctx$network$state_dict(), file.path(self$path, paste0("network", suffix, ".pt")))
      torch_save(self$ctx$optimizer$state_dict(), file.path(self$path, paste0("optimizer", suffix, ".pt")))
      if (!save_state) {
        return(invisible(NULL))
      }
      # what a later run needs on top of the network and the optimizer: the epoch to continue from
      # and the states of the other callbacks. These are plain R objects -- they are not
      # torch-serialized when a learner is marshaled either -- so they are saved with saveRDS(),
      # which unlike torch_save() keeps classes such as data.table intact.
      saveRDS(
        list(
          # the epoch this checkpoint belongs to, which is not necessarily `ctx$epoch`
          epoch     = suffix,
          callbacks = discard(map(self$ctx$callbacks, function(cb) cb$state_dict()), is.null)
        ),
        file.path(self$path, paste0("state", suffix, ".rds"))
      )
      invisible(NULL)
    }
  )
)

# Whether `path` exists and holds nothing.
is_empty_dir = function(path) {
  dir.exists(path) && !length(list.files(path, all.files = TRUE, no.. = TRUE))
}

# Whether it is safe for a CallbackSetCheckpoint to write into the existing folder `path`.
can_checkpoint_into = function(path) {
  is_empty_dir(path) ||
    (dir.exists(path) && length(list.files(path, pattern = "^(network|optimizer)[0-9]+\\.pt$|^state[0-9]+\\.rds$")) > 0L) # nolint
}

# The suffixes of the complete checkpoints in `path`, most recent first.
checkpoint_suffixes = function(path) {
  if (!dir.exists(path)) return(integer(0))
  suffixes = as.integer(gsub("^network|\\.pt$", "", list.files(path, pattern = "^network[0-9]+\\.pt$")))
  # a checkpoint is only usable if the matching optimizer was written as well, which is not the
  # case when a run was interrupted between the two
  sort(suffixes[file.exists(file.path(path, paste0("optimizer", suffixes, ".pt")))], decreasing = TRUE)
}

# The files of the most recent complete checkpoint in `path`, or NULL if there is none.
# `state` is NULL for a checkpoint that was written without one, see $on_exit().
latest_checkpoint = function(path) {
  suffixes = checkpoint_suffixes(path)
  if (!length(suffixes)) return(NULL)

  suffix = suffixes[1L]
  state = file.path(path, paste0("state", suffix, ".rds"))
  list(
    network   = file.path(path, paste0("network", suffix, ".pt")),
    optimizer = file.path(path, paste0("optimizer", suffix, ".pt")),
    state     = if (file.exists(state)) state,
    epoch     = suffix
  )
}

#' @include TorchCallback.R
mlr3torch_callbacks$add("checkpoint", function() {
  TorchCallback$new(
    callback_generator = CallbackSetCheckpoint,
    param_set = ps(
      path = p_uty(tags = c("train", "required")),
      freq = p_int(lower = 1L, tags = c("train", "required"))
    ),
    id = "checkpoint",
    label = "Checkpoint",
    man = "mlr3torch::mlr_callback_set.checkpoint"
  )
})
