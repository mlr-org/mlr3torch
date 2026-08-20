# Checkpoint Callback

Saves the optimizer, weights, and callback states every `freq` epochs as
well as the final state. This can be used to later continue a training
run via the `resume` parameter of
[`LearnerTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners_torch.md).

A folder holds the checkpoints of a single run, which is continued from
where it ended: training errors when `epochs` is less than the most
recent checkpoint in `path`, and a run never writes over a checkpoint
that is already there. When `epochs` is exactly that checkpoint, the run
in the folder is already finished: it is loaded and returned, and
nothing is written. Each file is checked again immediately before it is
written and an existing one is an error, so a second run writing into
the same folder is also caught when it started after this run did – the
check the folder gets before training cannot see it. The exception is a
checkpoint that was already half-written when this run started, which is
what a run killed mid-write leaves behind and which may be completed.

Checkpoints are written at the end of an epoch. For one written after
epoch `<n>`, three files are created in `path`:

- `network<n>.pt` :: The `$state_dict()` of the network.

- `optimizer<n>.pt` :: The `$state_dict()` of the optimizer. Next to
  them, written once with the first checkpoint, is `run.rds`: the task
  the folder's run trains on and the rows of its internal validation
  split, which a resuming run is checked against.

- `state<n>.rds` :: The epoch, the version of `mlr3torch` that wrote the
  checkpoint, as well as the `$state_dict()`s of the training run's
  other callbacks, so that a later run can continue e.g. the training
  history or the learning rate schedule. The class of each of those
  callbacks is recorded next to its state, which lets a resuming run
  notice that an id stands for a callback of another class instead of
  restoring the state of one into the other. Callbacks of the same class
  are indistinguishable to that check, and
  [`torch_callback()`](https://mlr3torch.mlr-org.com/dev/reference/torch_callback.md)
  names the class after the id, so a custom callback under a builtin's
  id is not caught by it. This file is written with
  [`saveRDS()`](https://rdrr.io/r/base/readRDS.html), so a callback
  state containing a `torch` tensor or module is written as an invalid
  pointer and errors when a resuming run uses it – see section
  *Inheriting* of
  [`CallbackSet`](https://mlr3torch.mlr-org.com/dev/reference/mlr_callback_set.md).

## Details

Saving the learner itself in the callback with a trained model is
impossible, as the model slot is set *after* the last callback step is
executed.

## Resuming

This callback is special because it enables resuming a training run. Its
own state is the folder it writes to, which
`learner$model$callbacks$<id>$path` reports – the only way to learn
where a `path` function sent a run, e.g. one fit of a
[`resample()`](https://mlr3.mlr-org.com/reference/resample.html). That
state is not part of a checkpoint and is not restored: a resuming run
writes where its own `path` says.

## Ordering

This callback has weight `Inf` and therefore runs last, so it captures
all the changes other callbacks made.

## See also

Other Callback:
[`TorchCallback`](https://mlr3torch.mlr-org.com/dev/reference/TorchCallback.md),
[`as_torch_callback()`](https://mlr3torch.mlr-org.com/dev/reference/as_torch_callback.md),
[`as_torch_callbacks()`](https://mlr3torch.mlr-org.com/dev/reference/as_torch_callbacks.md),
[`callback_set()`](https://mlr3torch.mlr-org.com/dev/reference/callback_set.md),
[`mlr3torch_callbacks`](https://mlr3torch.mlr-org.com/dev/reference/mlr3torch_callbacks.md),
[`mlr_callback_set`](https://mlr3torch.mlr-org.com/dev/reference/mlr_callback_set.md),
[`mlr_callback_set.progress`](https://mlr3torch.mlr-org.com/dev/reference/mlr_callback_set.progress.md),
[`mlr_callback_set.tb`](https://mlr3torch.mlr-org.com/dev/reference/mlr_callback_set.tb.md),
[`mlr_callback_set.unfreeze`](https://mlr3torch.mlr-org.com/dev/reference/mlr_callback_set.unfreeze.md),
[`mlr_context_torch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_context_torch.md),
[`t_clbk()`](https://mlr3torch.mlr-org.com/dev/reference/t_clbk.md),
[`torch_callback()`](https://mlr3torch.mlr-org.com/dev/reference/torch_callback.md)

## Super class

[`CallbackSet`](https://mlr3torch.mlr-org.com/dev/reference/mlr_callback_set.md)
-\> `CallbackSetCheckpoint`

## Public fields

- `weight`:

  (`numeric(1)`)  
  Always `Inf`, see section *Ordering*.

## Methods

### Public methods

- [`CallbackSetCheckpoint$new()`](#method-CallbackSetCheckpoint-initialize)

- [`CallbackSetCheckpoint$state_dict()`](#method-CallbackSetCheckpoint-state_dict)

- [`CallbackSetCheckpoint$on_begin()`](#method-CallbackSetCheckpoint-on_begin)

- [`CallbackSetCheckpoint$on_epoch_end()`](#method-CallbackSetCheckpoint-on_epoch_end)

- [`CallbackSetCheckpoint$on_end()`](#method-CallbackSetCheckpoint-on_end)

- [`CallbackSetCheckpoint$clone()`](#method-CallbackSetCheckpoint-clone)

Inherited methods

- [`CallbackSet$load_state_dict()`](https://mlr3torch.mlr-org.com/dev/reference/CallbackSet.html#method-load_state_dict)
- [`CallbackSet$print()`](https://mlr3torch.mlr-org.com/dev/reference/CallbackSet.html#method-print)

------------------------------------------------------------------------

### `CallbackSetCheckpoint$new()`

Creates a new instance of this
[R6](https://r6.r-lib.org/reference/R6Class.html) class.

#### Usage

    CallbackSetCheckpoint$new(path, freq)

#### Arguments

- `path`:

  (`character(1)` \| `function()`)  
  The path to a folder where the models are saved, or a function of no
  arguments returning it. The latter is especially useful to create
  unique directories during
  [`resample()`](https://mlr3.mlr-org.com/reference/resample.html) or
  [`benchmark()`](https://mlr3.mlr-org.com/reference/benchmark.html) per
  fit. The folder must be new, empty, or already contain checkpoints. A
  half-written checkpoint – what a run killed mid-write leaves behind –
  may be written over, since a resuming run continues from the newest
  complete one.

- `freq`:

  (`integer(1)`)  
  How often the model is saved, in epochs.

------------------------------------------------------------------------

### `CallbackSetCheckpoint$state_dict()`

Returns the folder this callback writes to, so that
`learner$model$callbacks$<id>$path` names it. This is the only place it
can be read off a trained learner when `path` is a `function()`, which
is called once per training run.

#### Usage

    CallbackSetCheckpoint$state_dict()

------------------------------------------------------------------------

### `CallbackSetCheckpoint$on_begin()`

Refuses to start when this run would not get past the checkpoint that is
already in `path`, or would write over the checkpoint of another run.

#### Usage

    CallbackSetCheckpoint$on_begin()

------------------------------------------------------------------------

### `CallbackSetCheckpoint$on_epoch_end()`

Saves the network and optimizer state dict. Does nothing if `freq` is
not met.

#### Usage

    CallbackSetCheckpoint$on_epoch_end()

------------------------------------------------------------------------

### `CallbackSetCheckpoint$on_end()`

Saves the final network and optimizer, unless the last epoch was already
saved.

#### Usage

    CallbackSetCheckpoint$on_end()

------------------------------------------------------------------------

### `CallbackSetCheckpoint$clone()`

The objects of this class are cloneable with this method.

#### Usage

    CallbackSetCheckpoint$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.

## Examples

``` r
cb = t_clbk("checkpoint", freq = 1)
task = tsk("iris")

pth = tempfile()
learner = lrn("classif.mlp", epochs = 3, batch_size = 1, callbacks = cb)
learner$param_set$set_values(cb.checkpoint.path = pth)

learner$train(task)

list.files(pth)
#>  [1] "network1.pt"   "network2.pt"   "network3.pt"   "optimizer1.pt"
#>  [5] "optimizer2.pt" "optimizer3.pt" "run.rds"       "state1.rds"   
#>  [9] "state2.rds"    "state3.rds"   

# continue training for 3 more epochs, starting from the last checkpoint
learner_resumed = lrn("classif.mlp", epochs = 6, batch_size = 1, resume = pth)
learner_resumed$train(task)
learner_resumed$model$epochs
#> [1] 6
```
