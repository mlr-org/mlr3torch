# Reduce On Plateau Learning Rate Scheduler

Reduces the learning rate when the first validation metric stops
improving for `patience` epochs. Wraps
[`torch::lr_reduce_on_plateau()`](https://torch.mlverse.org/docs/reference/lr_reduce_on_plateau.html)

## Super classes

[`CallbackSet`](https://mlr3torch.mlr-org.com/dev/reference/mlr_callback_set.md)
-\>
[`CallbackSetLRScheduler`](https://mlr3torch.mlr-org.com/dev/reference/mlr_callback_set.lr_scheduler.md)
-\> `CallbackSetLRSchedulerReduceOnPlateau`

## Methods

### Public methods

- [`CallbackSetLRSchedulerReduceOnPlateau$new()`](#method-CallbackSetLRSchedulerReduceOnPlateau-initialize)

- [`CallbackSetLRSchedulerReduceOnPlateau$clone()`](#method-CallbackSetLRSchedulerReduceOnPlateau-clone)

Inherited methods

- [`CallbackSet$load_state_dict()`](https://mlr3torch.mlr-org.com/dev/reference/CallbackSet.html#method-load_state_dict)
- [`CallbackSet$print()`](https://mlr3torch.mlr-org.com/dev/reference/CallbackSet.html#method-print)
- [`CallbackSet$state_dict()`](https://mlr3torch.mlr-org.com/dev/reference/CallbackSet.html#method-state_dict)
- [`CallbackSetLRScheduler$on_begin()`](https://mlr3torch.mlr-org.com/dev/reference/CallbackSetLRScheduler.html#method-on_begin)

------------------------------------------------------------------------

### `CallbackSetLRSchedulerReduceOnPlateau$new()`

Creates a new instance of this
[R6](https://r6.r-lib.org/reference/R6Class.html) class.

#### Usage

    CallbackSetLRSchedulerReduceOnPlateau$new(...)

#### Arguments

- `...`:

  (any)  
  The scheduler-specific initialization arguments.

------------------------------------------------------------------------

### `CallbackSetLRSchedulerReduceOnPlateau$clone()`

The objects of this class are cloneable with this method.

#### Usage

    CallbackSetLRSchedulerReduceOnPlateau$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.
