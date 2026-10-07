# Recurrent Layer

Base class for the recurrent layers
[`PipeOpTorchRNN`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_rnn.md),
[`PipeOpTorchLSTM`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_lstm.md)
and
[`PipeOpTorchGRU`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_gru.md).

## Parameters

See the respective child class.

## State

The state is the value calculated by the public method `shapes_out()`.

## Input and Output Channels

One input channel `"input"`, and the output channels described in the
child classes. For an explanation see
[`PipeOpTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch.md).

## See also

Other PipeOps:
[`mlr_pipeops_torch_model`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch_model.md)

## Super classes

[`mlr3pipelines::PipeOp`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html)
-\>
[`PipeOpTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch.md)
-\> `PipeOpTorchRecurrent`

## Methods

### Public methods

- [`PipeOpTorchRecurrent$new()`](#method-PipeOpTorchRecurrent-initialize)

- [`PipeOpTorchRecurrent$clone()`](#method-PipeOpTorchRecurrent-clone)

Inherited methods

- [`mlr3pipelines::PipeOp$help()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-help)
- [`mlr3pipelines::PipeOp$predict()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-predict)
- [`mlr3pipelines::PipeOp$print()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-print)
- [`mlr3pipelines::PipeOp$train()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-train)
- [`PipeOpTorch$shapes_out()`](https://mlr3torch.mlr-org.com/dev/reference/PipeOpTorch.html#method-shapes_out)

------------------------------------------------------------------------

### `PipeOpTorchRecurrent$new()`

Creates a new instance of this
[R6](https://r6.r-lib.org/reference/R6Class.html) class.

#### Usage

    PipeOpTorchRecurrent$new(id, type, param_set, param_vals = list())

#### Arguments

- `id`:

  (`character(1)`)  
  Identifier of the resulting object.

- `type`:

  (`character(1)`)  
  Which recurrent layer to build, one of `"rnn"`, `"lstm"` or `"gru"`.

- `param_set`:

  ([`ParamSet`](https://paradox.mlr-org.com/reference/ParamSet.html))  
  The parameter set.

- `param_vals`:

  ([`list()`](https://rdrr.io/r/base/list.html))  
  List of hyperparameter settings, overwriting the hyperparameter
  settings that would otherwise be set during construction.

------------------------------------------------------------------------

### `PipeOpTorchRecurrent$clone()`

The objects of this class are cloneable with this method.

#### Usage

    PipeOpTorchRecurrent$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.
