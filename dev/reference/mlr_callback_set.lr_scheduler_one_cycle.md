# OneCycle Learning Rate Scheduling Callback

Changes the learning rate based on the 1cycle learning rate policy.

Wraps
[`torch::lr_one_cycle()`](https://torch.mlverse.org/docs/reference/lr_one_cycle.html),
where the default values for `epochs` and `steps_per_epoch` are the
number of training epochs and the number of batches per epoch.

## Super classes

[`mlr3torch::CallbackSet`](https://mlr3torch.mlr-org.com/dev/reference/mlr_callback_set.md)
-\>
[`mlr3torch::CallbackSetLRScheduler`](https://mlr3torch.mlr-org.com/dev/reference/mlr_callback_set.lr_scheduler.md)
-\> `CallbackSetLRSchedulerOneCycle`

## Methods

### Public methods

- [`CallbackSetLRSchedulerOneCycle$new()`](#method-CallbackSetLRSchedulerOneCycle-new)

- [`CallbackSetLRSchedulerOneCycle$on_begin()`](#method-CallbackSetLRSchedulerOneCycle-on_begin)

- [`CallbackSetLRSchedulerOneCycle$clone()`](#method-CallbackSetLRSchedulerOneCycle-clone)

Inherited methods

- [`mlr3torch::CallbackSet$load_state_dict()`](https://mlr3torch.mlr-org.com/dev/reference/CallbackSet.html#method-load_state_dict)
- [`mlr3torch::CallbackSet$print()`](https://mlr3torch.mlr-org.com/dev/reference/CallbackSet.html#method-print)
- [`mlr3torch::CallbackSet$state_dict()`](https://mlr3torch.mlr-org.com/dev/reference/CallbackSet.html#method-state_dict)

------------------------------------------------------------------------

### Method `new()`

Creates a new instance of this
[R6](https://r6.r-lib.org/reference/R6Class.html) class.

#### Usage

    CallbackSetLRSchedulerOneCycle$new(...)

#### Arguments

- `...`:

  (any)  
  The scheduler-specific initialization arguments.

------------------------------------------------------------------------

### Method `on_begin()`

Creates the scheduler using the optimizer from the context

#### Usage

    CallbackSetLRSchedulerOneCycle$on_begin()

------------------------------------------------------------------------

### Method `clone()`

The objects of this class are cloneable with this method.

#### Usage

    CallbackSetLRSchedulerOneCycle$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.
