# Reduce On Plateau Learning Rate Scheduler

Reduces the learning rate when the first validation measure, i.e. the
first entry of the learner's `measures_valid`, stops improving for
`patience` evaluation steps. Wraps
[`torch::lr_reduce_on_plateau()`](https://torch.mlverse.org/docs/reference/lr_reduce_on_plateau.html).

Note that `patience` counts *evaluation steps*, not epochs: when the
learner's `eval_freq` is greater than `1`, the scheduler only steps in
the epochs in which the validation measures are evaluated, so `patience`
evaluation steps correspond to `patience * eval_freq` epochs.

Unless `mode` is set, it is derived from the measure: `"min"` if the
measure is minimized and `"max"` if it is maximized. Training fails when
`mode` contradicts the measure. When the learner has no validation
measure, e.g. in the final fit of an
[`AutoTuner`](https://mlr3tuning.mlr-org.com/reference/AutoTuner.html)
that tunes the epochs internally, a warning is given and the learning
rate is not reduced.

## Resuming

As for
[`CallbackSetLRScheduler`](https://mlr3torch.mlr-org.com/reference/mlr_callback_set.lr_scheduler.md).
For this schedule the restored state includes the best score seen so far
and how long it has been stagnating, so `patience` keeps counting across
runs instead of starting over.

## Super classes

[`CallbackSet`](https://mlr3torch.mlr-org.com/reference/mlr_callback_set.md)
-\>
[`CallbackSetLRScheduler`](https://mlr3torch.mlr-org.com/reference/mlr_callback_set.lr_scheduler.md)
-\> `CallbackSetLRSchedulerReduceOnPlateau`

## Methods

### Public methods

- [`CallbackSetLRSchedulerReduceOnPlateau$new()`](#method-CallbackSetLRSchedulerReduceOnPlateau-initialize)

- [`CallbackSetLRSchedulerReduceOnPlateau$on_begin()`](#method-CallbackSetLRSchedulerReduceOnPlateau-on_begin)

- [`CallbackSetLRSchedulerReduceOnPlateau$clone()`](#method-CallbackSetLRSchedulerReduceOnPlateau-clone)

Inherited methods

- [`CallbackSet$print()`](https://mlr3torch.mlr-org.com/reference/CallbackSet.html#method-print)
- [`CallbackSetLRScheduler$load_state_dict()`](https://mlr3torch.mlr-org.com/reference/CallbackSetLRScheduler.html#method-load_state_dict)
- [`CallbackSetLRScheduler$state_dict()`](https://mlr3torch.mlr-org.com/reference/CallbackSetLRScheduler.html#method-state_dict)

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

### `CallbackSetLRSchedulerReduceOnPlateau$on_begin()`

Creates the scheduler using the optimizer from the context, with the
`mode` that fits the first validation measure.

#### Usage

    CallbackSetLRSchedulerReduceOnPlateau$on_begin()

------------------------------------------------------------------------

### `CallbackSetLRSchedulerReduceOnPlateau$clone()`

The objects of this class are cloneable with this method.

#### Usage

    CallbackSetLRSchedulerReduceOnPlateau$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.
