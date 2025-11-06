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

## Super class

[`mlr3torch::CallbackSet`](https://mlr3torch.mlr-org.com/reference/mlr_callback_set.md)
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

- [`CallbackSetLRScheduler$new()`](#method-CallbackSetLRScheduler-new)

- [`CallbackSetLRScheduler$on_begin()`](#method-CallbackSetLRScheduler-on_begin)

- [`CallbackSetLRScheduler$clone()`](#method-CallbackSetLRScheduler-clone)

Inherited methods

- [`mlr3torch::CallbackSet$load_state_dict()`](https://mlr3torch.mlr-org.com/reference/CallbackSet.html#method-load_state_dict)
- [`mlr3torch::CallbackSet$print()`](https://mlr3torch.mlr-org.com/reference/CallbackSet.html#method-print)
- [`mlr3torch::CallbackSet$state_dict()`](https://mlr3torch.mlr-org.com/reference/CallbackSet.html#method-state_dict)

------------------------------------------------------------------------

### Method `new()`

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

### Method `on_begin()`

Creates the scheduler using the optimizer from the context

#### Usage

    CallbackSetLRScheduler$on_begin()

------------------------------------------------------------------------

### Method `clone()`

The objects of this class are cloneable with this method.

#### Usage

    CallbackSetLRScheduler$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.
