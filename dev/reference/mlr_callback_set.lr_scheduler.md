# Learning Rate Scheduling Callback

Changes the learning rate based on the schedule specified by a
[`torch::lr_scheduler`](https://torch.mlverse.org/docs/reference/lr_scheduler.html).

As of this writing, the following are available:

- [`torch::lr_cosine_annealing()`](https://torch.mlverse.org/docs/reference/lr_cosine_annealing.html)

- [`torch::lr_lambda()`](https://torch.mlverse.org/docs/reference/lr_lambda.html)

- [`torch::lr_multiplicative()`](https://torch.mlverse.org/docs/reference/lr_multiplicative.html)

- [`torch::lr_one_cycle()`](https://torch.mlverse.org/docs/reference/lr_one_cycle.html)
  (where the default values for `epochs` and `steps_per_epoch` are the
  number of training epochs and the number of batches per epoch)

- [`torch::lr_reduce_on_plateau()`](https://torch.mlverse.org/docs/reference/lr_reduce_on_plateau.html)

- [`torch::lr_step()`](https://torch.mlverse.org/docs/reference/lr_step.html)

- Custom schedulers defined with
  [`torch::lr_scheduler()`](https://torch.mlverse.org/docs/reference/lr_scheduler.html).

## Resuming

The state of the wrapped `torch` scheduler is stored and restored, so a
resumed run continues the schedule instead of starting it over. Creating
a scheduler resets the optimizer's learning rate to the one the schedule
started at, so the rate the restored schedule had reached is put back
afterwards.

That state contains the scheduler's configuration as well as its
progress, and restoring it overwrites what the resuming run was
configured with. Resuming with different scheduler arguments – or a
different `opt.lr`, which the schedule's base rates are derived from –
therefore silently continues the schedule of the checkpointed run.
Configure both runs the same way; a schedule cannot be changed halfway
through.

## Super class

[`CallbackSet`](https://mlr3torch.mlr-org.com/dev/reference/mlr_callback_set.md)
-\> `CallbackSetLRScheduler`

## Public fields

- `scheduler_fn`:

  (`lr_scheduler_generator`)  
  The `torch` function that creates a learning rate scheduler

- `scheduler`:

  (`LRScheduler`)  
  The learning rate scheduler wrapped by this callback

## Methods

### Public methods

- [`CallbackSetLRScheduler$new()`](#method-CallbackSetLRScheduler-initialize)

- [`CallbackSetLRScheduler$on_begin()`](#method-CallbackSetLRScheduler-on_begin)

- [`CallbackSetLRScheduler$state_dict()`](#method-CallbackSetLRScheduler-state_dict)

- [`CallbackSetLRScheduler$load_state_dict()`](#method-CallbackSetLRScheduler-load_state_dict)

- [`CallbackSetLRScheduler$clone()`](#method-CallbackSetLRScheduler-clone)

Inherited methods

- [`CallbackSet$print()`](https://mlr3torch.mlr-org.com/dev/reference/CallbackSet.html#method-print)

------------------------------------------------------------------------

### `CallbackSetLRScheduler$new()`

Creates a new instance of this
[R6](https://r6.r-lib.org/reference/R6Class.html) class.

#### Usage

    CallbackSetLRScheduler$new(.scheduler, step_on_epoch, ...)

#### Arguments

- `.scheduler`:

  (`lr_scheduler_generator`)  
  The `torch` scheduler generator (e.g.
  [`torch::lr_step`](https://torch.mlverse.org/docs/reference/lr_step.html)).

- `step_on_epoch`:

  (`logical(1)`)  
  Whether the scheduler steps after every epoch (otherwise every batch).

- `...`:

  (any)  
  The scheduler-specific initialization arguments.

------------------------------------------------------------------------

### `CallbackSetLRScheduler$on_begin()`

Creates the scheduler using the optimizer from the context

#### Usage

    CallbackSetLRScheduler$on_begin()

------------------------------------------------------------------------

### `CallbackSetLRScheduler$state_dict()`

Returns the state of the wrapped `torch` scheduler, so that a later run
can continue the schedule instead of starting it over. Returns `NULL` if
the scheduler was not created yet, i.e. before the training loop began.

#### Usage

    CallbackSetLRScheduler$state_dict()

------------------------------------------------------------------------

### `CallbackSetLRScheduler$load_state_dict()`

Loads the state of the wrapped `torch` scheduler.

#### Usage

    CallbackSetLRScheduler$load_state_dict(state_dict)

#### Arguments

- `state_dict`:

  (named [`list()`](https://rdrr.io/r/base/list.html))  
  The state dict as retrieved via `$state_dict()`.

------------------------------------------------------------------------

### `CallbackSetLRScheduler$clone()`

The objects of this class are cloneable with this method.

#### Usage

    CallbackSetLRScheduler$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.
