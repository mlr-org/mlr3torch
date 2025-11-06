# Base Class for Callbacks

Base class from which callbacks should inherit (see section
*Inheriting*). A callback set is a collection of functions that are
executed at different stages of the training loop. They can be used to
gain more control over the training process of a neural network without
having to write everything from scratch.

When used a in torch learner, the `CallbackSet` is wrapped in a
[`TorchCallback`](https://mlr3torch.mlr-org.com/reference/TorchCallback.md).
The latters parameter set represents the arguments of the
`CallbackSet`'s `$initialize()` method.

## Inheriting

For each available stage (see section *Stages*) a public method
`$on_<stage>()` can be defined. The evaluation context (a
[`ContextTorch`](https://mlr3torch.mlr-org.com/reference/mlr_context_torch.md))
can be accessed via `self$ctx`, which contains the current state of the
training loop. This context is assigned at the beginning of the training
loop and removed afterwards. Different stages of a callback can
communicate with each other by assigning values to `$self`.

*State*: To be able to store information in the `$model` slot of a
[`LearnerTorch`](https://mlr3torch.mlr-org.com/reference/mlr_learners_torch.md),
callbacks support a state API. You can overload the `$state_dict()`
public method to define what will be stored in
`learner$model$callbacks$<id>` after training finishes. This then also
requires to implement a `$load_state_dict(state_dict)` method that
defines how to load a previously saved callback state into a different
callback. Note that the `$state_dict()` should not include the parameter
values that were used to initialize the callback.

For creating custom callbacks, the function
[`torch_callback()`](https://mlr3torch.mlr-org.com/reference/torch_callback.md)
is recommended, which creates a `CallbackSet` and then wraps it in a
[`TorchCallback`](https://mlr3torch.mlr-org.com/reference/TorchCallback.md).
To create a `CallbackSet` the convenience function
[`callback_set()`](https://mlr3torch.mlr-org.com/reference/callback_set.md)
can be used. These functions perform checks such as that the stages are
not accidentally misspelled.

## Stages

- `begin` :: Run before the training loop begins.

- `epoch_begin` :: Run he beginning of each epoch.

- `batch_begin` :: Run before the forward call.

- `after_backward` :: Run after the backward call.

- `batch_end` :: Run after the optimizer step.

- `batch_valid_begin` :: Run before the forward call in the validation
  loop.

- `batch_valid_end` :: Run after the forward call in the validation
  loop.

- `valid_end` :: Run at the end of validation.

- `epoch_end` :: Run at the end of each epoch.

- `end` :: Run after last epoch.

- `exit` :: Run at last, using
  [`on.exit()`](https://rdrr.io/r/base/on.exit.html).

## Terminate Training

If training is to be stopped, it is possible to set the field
`$terminate` of
[`ContextTorch`](https://mlr3torch.mlr-org.com/reference/mlr_context_torch.md).
At the end of every epoch this field is checked and if it is `TRUE`,
training stops. This can for example be used to implement custom early
stopping.

## See also

Other Callback:
[`TorchCallback`](https://mlr3torch.mlr-org.com/reference/TorchCallback.md),
[`as_torch_callback()`](https://mlr3torch.mlr-org.com/reference/as_torch_callback.md),
[`as_torch_callbacks()`](https://mlr3torch.mlr-org.com/reference/as_torch_callbacks.md),
[`callback_set()`](https://mlr3torch.mlr-org.com/reference/callback_set.md),
[`mlr3torch_callbacks`](https://mlr3torch.mlr-org.com/reference/mlr3torch_callbacks.md),
[`mlr_callback_set.checkpoint`](https://mlr3torch.mlr-org.com/reference/mlr_callback_set.checkpoint.md),
[`mlr_callback_set.progress`](https://mlr3torch.mlr-org.com/reference/mlr_callback_set.progress.md),
[`mlr_callback_set.tb`](https://mlr3torch.mlr-org.com/reference/mlr_callback_set.tb.md),
[`mlr_callback_set.unfreeze`](https://mlr3torch.mlr-org.com/reference/mlr_callback_set.unfreeze.md),
[`mlr_context_torch`](https://mlr3torch.mlr-org.com/reference/mlr_context_torch.md),
[`t_clbk()`](https://mlr3torch.mlr-org.com/reference/t_clbk.md),
[`torch_callback()`](https://mlr3torch.mlr-org.com/reference/torch_callback.md)

## Public fields

- `ctx`:

  ([`ContextTorch`](https://mlr3torch.mlr-org.com/reference/mlr_context_torch.md)
  or `NULL`)  
  The evaluation context for the callback. This field should always be
  `NULL` except during the `$train()` call of the torch learner.

## Active bindings

- `stages`:

  ([`character()`](https://rdrr.io/r/base/character.html))  
  The active stages of this callback set.

## Methods

### Public methods

- [`CallbackSet$print()`](#method-CallbackSet-print)

- [`CallbackSet$state_dict()`](#method-CallbackSet-state_dict)

- [`CallbackSet$load_state_dict()`](#method-CallbackSet-load_state_dict)

- [`CallbackSet$clone()`](#method-CallbackSet-clone)

------------------------------------------------------------------------

### Method [`print()`](https://rdrr.io/r/base/print.html)

Prints the object.

#### Usage

    CallbackSet$print(...)

#### Arguments

- `...`:

  (any)  
  Currently unused.

------------------------------------------------------------------------

### Method `state_dict()`

Returns information that is kept in the the
[`LearnerTorch`](https://mlr3torch.mlr-org.com/reference/mlr_learners_torch.md)'s
state after training. This information should be loadable into the
callback using `$load_state_dict()` to be able to continue training.
This returns `NULL` by default.

#### Usage

    CallbackSet$state_dict()

------------------------------------------------------------------------

### Method [`load_state_dict()`](https://torch.mlverse.org/docs/reference/load_state_dict.html)

Loads the state dict into the callback to continue training.

#### Usage

    CallbackSet$load_state_dict(state_dict)

#### Arguments

- `state_dict`:

  (any)  
  The state dict as retrieved via `$state_dict()`.

------------------------------------------------------------------------

### Method `clone()`

The objects of this class are cloneable with this method.

#### Usage

    CallbackSet$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.
