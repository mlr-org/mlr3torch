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
#' Training can be continued from such a checkpoint -- also in a new R session -- via the `path`
#' parameter of [`LearnerTorch`], see the example below.
#'
#' @details
#' Saving the learner itself in the callback with a trained model is impossible,
#' as the model slot is set *after* the last callback step is executed.
#'
#' @param path (`character(1)` | `function()`)\cr
#'   The path to a folder where the models are saved, or a function of no arguments returning it.
#'   The latter is especially useful to create unique directories during `resample()` or `benchmark()`
#'   per fit.
#'   The folder must be new, empty, or already contain checkpoints, so that a checkpoint never
#'   overwrites unrelated data.
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
    #' @param weight (`numeric(1)`)\cr
    #'   Controls when this callback is called within a stage, see section *Ordering* of
    #'   [`CallbackSet`].
    #'   Defaults to `Inf`, so that this callback runs after the other callbacks and hence saves the
    #'   network and optimizer as they are at the end of the stage.
    #'   The only exception is the restore of `restore_best_weights`, which happens afterwards, so
    #'   a checkpoint always holds the network as training left it.
    initialize = function(path, freq, weight = Inf) {
      self$weight = assert_number(weight)
      self$freq = assert_int(freq, lower = 1L)
      # a function is evaluated once per training run, which is what makes resampling, benchmarking
      # and tuning possible: every iteration can compute a path of its own
      if (is.function(path)) {
        path = assert_string(path(), .var.name = "return value of the `path` function")
      }
      # We can either start in a new folder or continue an already existing checkpoint
      self$path = if (can_checkpoint_into(path)) path else assert_path_for_output(path)
      if (!dir.exists(path)) {
        dir.create(path, recursive = TRUE)
      }
    },
    #' @description
    #' Refuses to start when this run would write over the checkpoint of another run.
    on_begin = function() {
      # the epoch this run starts from: 0, or the epoch a resumed checkpoint left off at
      private$.start_epoch = self$ctx$epoch
      # A run never writes the same epoch twice, so a complete checkpoint under an epoch this run
      # is going to write belongs to a different run and is not ours to destroy. A run continuing
      # an earlier one writes epochs that one does not have, so it does not collide. Checking here
      # rather than in $.save() means the folder is not half rewritten before this is noticed.
      planned = seq_len(self$ctx$total_epochs)
      clash = intersect(planned[planned > private$.start_epoch], checkpoint_files(self$path)$complete)
      if (length(clash)) {
        stopf("Checkpoint(s) %s in '%s' were written by another run, and this one would write over them. Use a fresh folder, or continue that run instead of starting it over.", # nolint
          paste0(clash, collapse = ", "), self$path)
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
      if (self$ctx$epoch == private$.start_epoch) {
        # this run trained no epochs of its own, so the checkpoint it resumed is the current one
        return(NULL)
      }
      if (self$ctx$epoch == 0L || self$ctx$epoch %% self$freq == 0) {
        # nothing was trained, or the last epoch was already saved
        return(NULL)
      }
      private$.save(self$ctx$epoch)
    }
  ),
  private = list(
    .start_epoch = 0L,
    .save = function(suffix) {
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

# The checkpoints in `path`, split into those that can be read and those that cannot. Both the
# reading and the writing side go through this, so they cannot disagree on what "exists" means:
# a checkpoint that is too incomplete to resume from is also one that may be written over.
checkpoint_files = function(path) {
  none = list(complete = integer(0), incomplete = integer(0))
  if (!dir.exists(path)) return(none)
  suffixes = as.integer(gsub("^network|\\.pt$", "", list.files(path, pattern = "^network[0-9]+\\.pt$")))
  # paste0() recycles a zero-length suffix to "", which would look for 'optimizer.pt'
  if (!length(suffixes)) return(none)
  complete = file.exists(file.path(path, paste0("optimizer", suffixes, ".pt"))) &
    file.exists(file.path(path, paste0("state", suffixes, ".rds")))
  list(
    complete = sort(suffixes[complete], decreasing = TRUE),
    incomplete = sort(suffixes[!complete])
  )
}

checkpoint_suffixes = function(path) {
  files = checkpoint_files(path)
  if (length(files$incomplete)) {
    # silently skipping these would look like the run simply got less far than it did
    warningf("Ignoring incomplete checkpoint(s) %s in '%s', which are missing an optimizer or a state file. This is what a run killed while writing a checkpoint leaves behind.", # nolint
      paste0(files$incomplete, collapse = ", "), path)
  }
  files$complete
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

# the path is either given directly or computed once per training run
check_checkpoint_path = function(x) {
  if (is.function(x)) check_function(x, nargs = 0L) else check_string(x)
}

#' @include TorchCallback.R
mlr3torch_callbacks$add("checkpoint", function() {
  TorchCallback$new(
    callback_generator = CallbackSetCheckpoint,
    param_set = ps(
      path = p_uty(tags = c("train", "required"), custom_check = check_checkpoint_path),
      freq = p_int(lower = 1L, tags = c("train", "required"))
    ),
    id = "checkpoint",
    label = "Checkpoint",
    man = "mlr3torch::mlr_callback_set.checkpoint"
  )
})
