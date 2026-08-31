# Checkpoint Callback

Saves the optimizer, weights, and callback states every `freq` epochs as
well as the final state. This can be used to later continue a training
run via the `resume` parameter of
[`LearnerTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners_torch.md).

Checkpoints are written at the end of an epoch. For one written after
epoch `<n>`, three files are created in `path`:

- `network<n>.pt` :: The `$state_dict()` of the network.

- `optimizer<n>.pt` :: The `$state_dict()` of the optimizer.

- `state<n>.rds` :: The epoch, the version of `mlr3torch` that wrote the
  checkpoint, the `$state_dict()`s of the training run's other
  callbacks, so that a later run can continue, as well as some other
  information. Additionally, there is `run.rds` which contains some
  additioanl global meta information.

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

Returns the folder this callback writes to so it can be accessed from
the learner when the `path` was a function.

#### Usage

    CallbackSetCheckpoint$state_dict()

------------------------------------------------------------------------

### `CallbackSetCheckpoint$on_begin()`

Checks whether the checkpoint path is valid.

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
