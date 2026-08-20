# OneCycle Learning Rate Scheduling Callback

Changes the learning rate based on the 1cycle learning rate policy.

Wraps
[`torch::lr_one_cycle()`](https://torch.mlverse.org/docs/reference/lr_one_cycle.html),
where the default values for `epochs` and `steps_per_epoch` are the
number of training epochs and the number of batches per epoch.

## Resuming

As for
[`CallbackSetLRScheduler`](https://mlr3torch.mlr-org.com/dev/reference/mlr_callback_set.lr_scheduler.md),
with one additional restriction: the 1cycle policy is defined over the
total number of steps of the run, so a resumed run must be configured
for the same number of steps as the one that wrote the checkpoint, i.e.
the same `epochs` and the same number of batches per epoch. A run that
is not errors before its first epoch rather than somewhere in the
middle.

## Super classes

[`CallbackSet`](https://mlr3torch.mlr-org.com/dev/reference/mlr_callback_set.md)
-\>
[`CallbackSetLRScheduler`](https://mlr3torch.mlr-org.com/dev/reference/mlr_callback_set.lr_scheduler.md)
-\> `CallbackSetLRSchedulerOneCycle`

## Methods

### Public methods

- [`CallbackSetLRSchedulerOneCycle$new()`](#method-CallbackSetLRSchedulerOneCycle-initialize)

- [`CallbackSetLRSchedulerOneCycle$on_begin()`](#method-CallbackSetLRSchedulerOneCycle-on_begin)

- [`CallbackSetLRSchedulerOneCycle$clone()`](#method-CallbackSetLRSchedulerOneCycle-clone)

Inherited methods

- [`CallbackSet$print()`](https://mlr3torch.mlr-org.com/dev/reference/CallbackSet.html#method-print)
- [`CallbackSetLRScheduler$load_state_dict()`](https://mlr3torch.mlr-org.com/dev/reference/CallbackSetLRScheduler.html#method-load_state_dict)
- [`CallbackSetLRScheduler$state_dict()`](https://mlr3torch.mlr-org.com/dev/reference/CallbackSetLRScheduler.html#method-state_dict)

------------------------------------------------------------------------

### `CallbackSetLRSchedulerOneCycle$new()`

Creates a new instance of this
[R6](https://r6.r-lib.org/reference/R6Class.html) class.

#### Usage

    CallbackSetLRSchedulerOneCycle$new(...)

#### Arguments

- `...`:

  (any)  
  The scheduler-specific initialization arguments.

------------------------------------------------------------------------

### `CallbackSetLRSchedulerOneCycle$on_begin()`

Creates the scheduler using the optimizer from the context

#### Usage

    CallbackSetLRSchedulerOneCycle$on_begin()

------------------------------------------------------------------------

### `CallbackSetLRSchedulerOneCycle$clone()`

The objects of this class are cloneable with this method.

#### Usage

    CallbackSetLRSchedulerOneCycle$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.
