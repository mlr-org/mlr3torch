# Merge Operation

Base class for merge operations such as addition
([`PipeOpTorchMergeSum`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_merge_sum.md)),
multiplication
([`PipeOpTorchMergeProd`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_merge_prod.md)
or concatenation
([`PipeOpTorchMergeCat`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_merge_cat.md)).

## Parameters

See the respective child class.

## State

The state is the value calculated by the public method `shapes_out()`.

## Input and Output Channels

`PipeOpTorchMerge`s has either a *vararg* input channel if the
constructor argument `innum` is not set, or input channels `"input1"`,
..., `"input<innum>"`. There is one output channel `"output"`. For an
explanation see
[`PipeOpTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch.md).

## Internals

Per default, the `private$.shapes_out()` method outputs the shape that
the inputs broadcast to. There are two things to be aware of:

1.  Broadcasting is generalized to unknown (`NA`) sizes: per dimension a
    known size that is not 1 wins, and the result is only unknown when
    every input is either unknown or 1, because an unknown size may turn
    out to be greater than 1 and would then determine the size.

2.  Tensors are expected to have the same number of dimensions, i.e.
    missing dimensions are not filled with 1s. The reason is that the
    first dimension should be the batch dimension. This private method
    can be overwritten by
    [`PipeOpTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch.md)s
    inheriting from this class.

## Super classes

[`mlr3pipelines::PipeOp`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html)
-\>
[`PipeOpTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch.md)
-\> `PipeOpTorchMerge`

## Methods

### Public methods

- [`PipeOpTorchMerge$new()`](#method-PipeOpTorchMerge-initialize)

- [`PipeOpTorchMerge$clone()`](#method-PipeOpTorchMerge-clone)

Inherited methods

- [`mlr3pipelines::PipeOp$help()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-help)
- [`mlr3pipelines::PipeOp$predict()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-predict)
- [`mlr3pipelines::PipeOp$print()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-print)
- [`mlr3pipelines::PipeOp$train()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-train)
- [`PipeOpTorch$shapes_out()`](https://mlr3torch.mlr-org.com/dev/reference/PipeOpTorch.html#method-shapes_out)

------------------------------------------------------------------------

### `PipeOpTorchMerge$new()`

Creates a new instance of this
[R6](https://r6.r-lib.org/reference/R6Class.html) class.

#### Usage

    PipeOpTorchMerge$new(
      id,
      module_generator,
      param_set = ps(),
      innum = 0,
      param_vals = list()
    )

#### Arguments

- `id`:

  (`character(1)`)  
  Identifier of the resulting object.

- `module_generator`:

  (`nn_module_generator`)  
  The torch module generator.

- `param_set`:

  ([`ParamSet`](https://paradox.mlr-org.com/reference/ParamSet.html))  
  The parameter set.

- `innum`:

  (`integer(1)`)  
  The number of inputs. Default is 0 which means there is one *vararg*
  input channel.

- `param_vals`:

  ([`list()`](https://rdrr.io/r/base/list.html))  
  List of hyperparameter settings, overwriting the hyperparameter
  settings that would otherwise be set during construction.

------------------------------------------------------------------------

### `PipeOpTorchMerge$clone()`

The objects of this class are cloneable with this method.

#### Usage

    PipeOpTorchMerge$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.
