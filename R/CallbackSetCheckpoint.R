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
#' * `state<n>.rds` :: The epoch, the version of `mlr3torch` that wrote the checkpoint, as well as
#'   the `$state_dict()`s of the other callbacks of the training run, so that a later run can
#'   continue e.g. the training history or the learning rate schedule.
#'   Resuming from a checkpoint written by a different version of `mlr3torch` warns, because what a
#'   callback stores in its state dict is up to the callback and may change between releases.
#'
#' A checkpoint counts as complete only if all three files are present, so a run that was killed
#' while writing one of them falls back to the previous checkpoint rather than to a partial one.
#' Reading a folder that holds such a partial checkpoint warns, as skipping it silently would look
#' like the run simply got less far than it did.
#'
#' `path` may already contain checkpoints, which is what a run continuing an earlier one writes
#' into: it writes epochs that folder does not have yet, so nothing of the earlier run is touched.
#' A run that would write over a checkpoint of another run errors instead, so a folder is never
#' silently left holding the epochs of two different trainings.
#' An incomplete checkpoint is replaced rather than protected -- it cannot be resumed from anyway,
#' and refusing would mean a run killed while writing could never be restarted into its own folder.
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
CallbackSetCheckpoint = R6Class("CallbackSetCheckpoint",
  inherit = CallbackSet,
  lock_objects = FALSE,
  # TODO: This should also save the learner itself
  public = list(
    #' @field weight (`numeric(1)`)\cr
    #'   `Inf`, so that this callback runs after all others and hence saves the network and
    #'   optimizer as they are at the end of the stage, see section *Ordering* of [`CallbackSet`].
    weight = Inf,
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(path, freq) {
      self$freq = assert_int(freq, lower = 1L)
      # We can either start in a new folder or continue an already existing checkpoint
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
      # NOT on_exit, because we only write when the epoch ran successfully
      if (self$ctx$epoch == 0L || self$ctx$epoch %% self$freq == 0) {
        # nothing was trained, or the last epoch was already saved
        return(NULL)
      }
      private$.save(self$ctx$epoch)
    }
  ),
  private = list(
    .save = function(suffix) {
      # A run never writes the same epoch twice, so a complete checkpoint under this number belongs
      # to a different run and is not ours to destroy. A run continuing an earlier one writes epochs
      # that one does not have, so it never gets here; an incomplete checkpoint has no state file
      # and is unusable anyway, so replacing that is allowed -- otherwise a run killed while writing
      # could never be restarted into its own folder.
      if (file.exists(file.path(self$path, paste0("state", suffix, ".rds")))) {
        stopf("Checkpoint of epoch %i already exists in '%s' and was written by another run. Use a fresh folder, or continue that run instead of starting it over.", # nolint
          suffix, self$path)
      }
      torch_save(self$ctx$network$state_dict(), file.path(self$path, paste0("network", suffix, ".pt")))
      torch_save(self$ctx$optimizer$state_dict(), file.path(self$path, paste0("optimizer", suffix, ".pt")))
      saveRDS(
        list(
          # the epoch this checkpoint belongs to, i.e. the one that just ran to its end
          epoch = suffix,
          # what wrote this checkpoint. What a callback state dict contains is up to the callback,
          # so it can change between releases; recording the version lets a later run say so instead
          # of failing somewhere inside `$load_state_dict()`.
          version = as.character(utils::packageVersion("mlr3torch")),
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

checkpoint_suffixes = function(path) {
  if (!dir.exists(path)) return(integer(0))
  suffixes = as.integer(gsub("^network|\\.pt$", "", list.files(path, pattern = "^network[0-9]+\\.pt$")))
  # paste0() recycles a zero-length suffix to "", which would look for 'optimizer.pt'
  if (!length(suffixes)) return(integer(0))
  complete = file.exists(file.path(path, paste0("optimizer", suffixes, ".pt"))) &
    file.exists(file.path(path, paste0("state", suffixes, ".rds")))
  if (!all(complete)) {
    # silently skipping these would look like the run simply got less far than it did
    warningf("Ignoring incomplete checkpoint(s) %s in '%s', which are missing an optimizer or a state file. This is what a run killed while writing a checkpoint leaves behind.", # nolint
      paste0(sort(suffixes[!complete]), collapse = ", "), path)
  }
  sort(suffixes[complete], decreasing = TRUE)
}

read_checkpoint_state = function(file) {
  state = readRDS(file)
  current = as.character(utils::packageVersion("mlr3torch"))
  if (!identical(state$version, current)) {
    written_by = if (is.null(state$version)) {
      "a version of mlr3torch from before checkpoints recorded one"
    } else {
      sprintf("mlr3torch %s", state$version)
    }
    warningf("Checkpoint '%s' was written by %s, but mlr3torch %s is loaded. Resuming from it may fail or restore an incomplete state, because what a callback stores in its state dict is version specific.", # nolint
      file, written_by, current)
  }
  state
}

# The files of the most recent complete checkpoint in `path`, or NULL if there is none.
latest_checkpoint = function(path) {
  suffixes = checkpoint_suffixes(path)
  if (!length(suffixes)) return(NULL)

  suffix = suffixes[1L]
  list(
    network   = file.path(path, paste0("network", suffix, ".pt")),
    optimizer = file.path(path, paste0("optimizer", suffix, ".pt")),
    state     = file.path(path, paste0("state", suffix, ".rds")),
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
