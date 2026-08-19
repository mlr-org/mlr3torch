#' @title Checkpoint Callback
#'
#' @name mlr_callback_set.checkpoint
#'
#' @description
#' Saves the optimizer, weights, and callback states every `freq` epochs as well as the final state.
#' This can be used to later continue a training run via the `resume` parameter of [`LearnerTorch`].
#'
#' A folder holds the checkpoints of a single run, which is continued from where it ended: training
#' errors when `epochs` is less than the most recent checkpoint in `path`, and a run never
#' writes over a checkpoint that is already there.
#' When `epochs` is exactly that checkpoint, the run in the folder is already finished: it is
#' loaded and returned, and nothing is written.
#' Each file is checked again immediately before it is written and an existing one is an error, so
#' a second run writing into the same folder is also caught when it started after this run did --
#' the check the folder gets before training cannot see it.
#' The exception is a checkpoint that was already half-written when this run started, which is what
#' a run killed mid-write leaves behind and which may be completed.
#'
#' Checkpoints are written at the end of an epoch. For one written after epoch `<n>`, three files
#' are created in `path`:
#' * `network<n>.pt` :: The `$state_dict()` of the network.
#' * `optimizer<n>.pt` :: The `$state_dict()` of the optimizer.
#' Next to them, written once with the first checkpoint, is `run.rds`: the task the folder's run
#' trains on and the rows of its internal validation split, which a resuming run is checked against.
#' * `state<n>.rds` :: The epoch, the version of `mlr3torch` that wrote the checkpoint, as well as
#'   the `$state_dict()`s of the training run's other callbacks, so that a later run can
#'   continue e.g. the training history or the learning rate schedule.
#'   The class of each of those callbacks is recorded next to its state, which lets a resuming run
#'   notice that an id stands for a callback of another class instead of restoring the state of one
#'   into the other.
#'   Callbacks of the same class are indistinguishable to that check, and [`torch_callback()`] names
#'   the class after the id, so a custom callback under a builtin's id is not caught by it.
#'
#' @section Resuming:
#' This callback is special because it enables resuming a training run.
#' Its own state is the folder it writes to, which `learner$model$callbacks$<id>$path` reports --
#' the only way to learn where a `path` function sent a run, e.g. one fit of a `resample()`.
#' That state is not part of a checkpoint and is not restored: a resuming run writes where its own
#' `path` says.
#' @section Ordering:
#' This callback has weight `Inf` and therefore runs last, so it captures all the changes other
#' callbacks made.
#'
#' @details
#' Saving the learner itself in the callback with a trained model is impossible,
#' as the model slot is set *after* the last callback step is executed.
#'
#' @param path (`character(1)` | `function()`)\cr
#'   The path to a folder where the models are saved, or a function of no arguments returning it.
#'   The latter is especially useful to create unique directories during `resample()` or `benchmark()`
#'   per fit.
#'   The folder must be new, empty, or already contain checkpoints.
#'   A half-written checkpoint -- what a run killed mid-write leaves behind -- may be written over,
#'   since a resuming run continues from the newest complete one.
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
#' learner_resumed = lrn("classif.mlp", epochs = 6, batch_size = 1, resume = pth)
#' learner_resumed$train(task)
#' learner_resumed$model$epochs
CallbackSetCheckpoint = R6Class("CallbackSetCheckpoint",
  inherit = CallbackSet,
  lock_objects = FALSE,
  # TODO: This should also save the learner itself
  public = list(
    #' @field weight (`numeric(1)`)\cr
    #'   Always `Inf`, see section *Ordering*.
    weight = Inf,
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(path, freq) {
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
    #' Returns the folder this callback writes to, so that `learner$model$callbacks$<id>$path`
    #' names it. This is the only place it can be read off a trained learner when `path` is a
    #' `function()`, which is called once per training run.
    state_dict = function() {
      list(path = self$path)
    },
    #' @description
    #' Refuses to start when this run would not get past the checkpoint that is already in `path`,
    #' or would write over the checkpoint of another run.
    on_begin = function() {
      # the epoch this run starts from: 0, or the epoch a resumed checkpoint left off at
      private$.start_epoch = self$ctx$epoch
      files = checkpoint_files(self$path)
      complete = files$complete
      # a checkpoint that is already half-written is what a run killed mid-write leaves behind, and
      # the only thing this run is allowed to write over, see $.save()
      private$.overwritable = files$incomplete
      # The most recent checkpoint is where the run in this folder ended, which is also where a
      # later resume picks it up again. A run that does not train past it therefore leaves the
      # folder holding the epochs of two runs while still resuming into the earlier one.
      # A run that trains nothing at all -- one continuing a checkpoint that is already at `epochs`
      # -- writes nothing either, so it leaves the folder exactly as it found it.
      trains_something = self$ctx$total_epochs > private$.start_epoch
      if (length(complete) && trains_something && self$ctx$total_epochs <= complete[1L]) {
        stopf("The most recent checkpoint in '%s' is at epoch %i, but 'epochs' is %i, so this run would not get past it. A folder holds the checkpoints of one run, and a run continuing it has to train beyond its last epoch: use a fresh folder, or set 'epochs' to more than %i.", # nolint
          self$path, complete[1L], self$ctx$total_epochs, complete[1L])
      }
      # A run never writes the same epoch twice, so a complete checkpoint under an epoch this run
      # is going to write belongs to a different run and is not ours to destroy. A run continuing
      # an earlier one writes epochs that one does not have, so it does not collide. Checking here
      # rather than in $.save() means the folder is not half rewritten before this is noticed.
      planned = seq_len(self$ctx$total_epochs)
      clash = intersect(planned[planned > private$.start_epoch], complete)
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
      # NOT on_exit, because we only write when the epoch ran successfully.
      # Nothing to do when this run trained no epochs of its own -- whatever it resumed is then the
      # current checkpoint -- or when `freq` already saved the epoch it ended on.
      if (self$ctx$epoch == private$.start_epoch || self$ctx$epoch %% self$freq == 0) {
        return(NULL)
      }
      private$.save(self$ctx$epoch)
    }
  ),
  private = list(
    .start_epoch = 0L,
    .overwritable = integer(0),
    .save = function(suffix) {
      network_file = file.path(self$path, paste0("network", suffix, ".pt"))
      optimizer_file = file.path(self$path, paste0("optimizer", suffix, ".pt"))
      state_file = file.path(self$path, paste0("state", suffix, ".rds"))
      # $on_begin() reads the folder once, before training, so it cannot see a file that appears
      # while this run is training -- which is what a second run writing into the same folder looks
      # like. Checking again right before the write catches that: every epoch this run writes is
      # one that nothing had written when it started, apart from a half-written checkpoint it may
      # complete. Checking all three files before writing any of them keeps the folder as it was.
      if (suffix %nin% private$.overwritable) {
        clash = keep(c(network_file, optimizer_file, state_file), file.exists)
        if (length(clash)) {
          stopf("Refusing to write over %s, which appeared in '%s' while this run was training: another run is writing into the same folder. A folder holds the checkpoints of one run, so give every run one of its own, e.g. by setting `path` to a function returning a new folder per run.", # nolint
            paste0("'", basename(clash), "'", collapse = ", "), self$path)
        }
      }
      # What this run trains on does not vary per epoch, so it is written once per folder rather
      # than into every state file. A folder holds one run and its continuations, which have to
      # agree on it anyway -- that is what $assert_resumable_task() checks on the reading side.
      run_file = file.path(self$path, "run.rds")
      if (!file.exists(run_file)) {
        saveRDS(
          list(
            task_id = self$ctx$task_train$id,
            valid_row_ids = if (!is.null(self$ctx$task_valid)) self$ctx$task_valid$row_ids
          ),
          run_file
        )
      }
      torch_save(self$ctx$network$state_dict(), network_file)
      torch_save(self$ctx$optimizer$state_dict(), optimizer_file)
      # A checkpoint callback's own state is the folder it writes to, which a resuming run does not
      # take over -- it writes where its own `path` says. So it belongs in `learner$model$callbacks`
      # but not in a file that exists to be restored.
      resumable = discard(self$ctx$callbacks, function(cb) inherits(cb, "CallbackSetCheckpoint"))
      states = discard(map(resumable, function(cb) cb$state_dict()), is.null)
      saveRDS(
        list(
          epoch = suffix,
          # the scores the model reports as its `internal_valid_scores`, so that a run continuing a
          # checkpoint that is already at `epochs` can report them without an epoch of its own
          valid_scores = self$ctx$last_scores_valid,
          # counted up by the training loop rather than derived, so a resumed run can only continue
          # the count if the checkpoint carries it
          global_step = self$ctx$global_step,
          version = as.character(utils::packageVersion("mlr3torch")),
          callbacks = states,
          classes = map_chr(resumable[names(states)], function(cb) class(cb)[[1L]])
        ),
        state_file
      )
      invisible(NULL)
    }
  )
)

# Whether `path` exists and holds nothing.
is_empty_dir = function(path) {
  dir.exists(path) && !length(list.files(path, all.files = TRUE, no.. = TRUE))
}

# Whether it is safe for a CallbackSetCheckpoint to write into the existing folder `path`, i.e.
# whether it is empty or holds a checkpoint folder whose newest epoch is complete. A newest epoch
# that is half-written is refused rather than replaced: it is what a resuming run would continue
# from, so a folder is only ever handed to a new run in a state that can be resumed.
can_checkpoint_into = function(path) {
  # An incomplete checkpoint is written over rather than protected: a resuming run continues from
  # the newest *complete* one, so a half-written epoch is not what anything reads, and refusing it
  # would mean a run killed while writing could never be restarted into its own folder.
  # $on_begin() still refuses to write over any complete checkpoint of another run.
  is_empty_dir(path) ||
    (dir.exists(path) && length(list.files(path, pattern = "^(network|optimizer)[0-9]+\\.pt$|^state[0-9]+\\.rds$|^run\\.rds$")) > 0L) # nolint
}

# The checkpoints in `path`, split into those that can be read and those that cannot. Both the
# reading and the writing side go through this, so they cannot disagree on what "exists" means:
# a checkpoint that is too incomplete to resume from is also one that may be written over.
checkpoint_files = function(path) {
  none = list(complete = integer(0), incomplete = integer(0))
  if (!dir.exists(path)) return(none)
  # every epoch any of the three files exists for, so that a leftover is seen whichever of them the
  # interrupted run had already written
  files = list.files(path, pattern = "^(network|optimizer)[0-9]+\\.pt$|^state[0-9]+\\.rds$")
  suffixes = unique(as.integer(gsub("^(network|optimizer|state)|\\.(pt|rds)$", "", files)))
  # paste0() recycles a zero-length suffix to "", which would look for 'optimizer.pt'
  if (!length(suffixes)) return(none)
  complete = file.exists(file.path(path, paste0("network", suffixes, ".pt"))) &
    file.exists(file.path(path, paste0("optimizer", suffixes, ".pt"))) &
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
    # the class declares the same weight, this makes it visible without generating the callback,
    # which is not possible without values for the required `path` and `freq`
    weight = Inf,
    man = "mlr3torch::mlr_callback_set.checkpoint"
  )
})
