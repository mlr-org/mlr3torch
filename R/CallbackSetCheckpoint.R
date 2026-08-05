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
#' A checkpoint is only ever written for an epoch that ran to its end, so `network<n>.pt` is always
#' the network at the *end* of epoch `n`.
#' If training fails in the middle of an epoch, nothing further is written -- the network and the
#' optimizer have already been updated by the batches of that epoch which did run, so they are at no
#' epoch boundary -- and the run keeps the checkpoints that `freq` had written before the error.
#'
#' Ending a run early is unaffected by this.
#' `ctx$terminate` is only acted on once the epoch has finished, so early stopping and any callback
#' that sets it leave the network at an epoch boundary and the last epoch is written as usual.
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
      if (self$ctx$epoch %% self$freq != 0) {
        return(NULL)
      }
      private$.save(self$ctx$epoch)
    },
    #' @description
    #' Saves the final network and optimizer, unless the last epoch was already saved.
    on_end = function() {
      # the 'end' stage rather than 'exit': it is only reached once the training loop is through,
      # so the network is at an epoch boundary here. 'exit' also runs when an error ended the run
      # in the middle of an epoch, where the batches that did run have already updated the network
      # and the optimizer -- saving those under the number of the last complete epoch would label a
      # half-trained state as a finished one.
      if (self$ctx$epoch == 0L || self$ctx$epoch %% self$freq == 0) {
        # nothing was trained, or the last epoch was already saved
        return(NULL)
      }
      private$.save(self$ctx$epoch)
    }
  ),
  private = list(
    .save = function(suffix) {
      torch_save(self$ctx$network$state_dict(), file.path(self$path, paste0("network", suffix, ".pt")))
      torch_save(self$ctx$optimizer$state_dict(), file.path(self$path, paste0("optimizer", suffix, ".pt")))
      # what a later run needs on top of the network and the optimizer: the epoch to continue from
      # and the states of the other callbacks. These are plain R objects -- they are not
      # torch-serialized when a learner is marshaled either -- so they are saved with saveRDS(),
      # which unlike torch_save() keeps classes such as data.table intact.
      saveRDS(
        list(
          # the epoch this checkpoint belongs to, i.e. the one that just ran to its end
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
