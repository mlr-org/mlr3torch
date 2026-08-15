# Torch Entry Point for Categorical Features

Ingress PipeOp that represents a categorical
([`factor()`](https://rdrr.io/r/base/factor.html),
[`ordered()`](https://rdrr.io/r/base/factor.html) and
[`logical()`](https://rdrr.io/r/base/logical.html)) entry point to a
torch network.

## Parameters

- `select` :: `logical(1)`  
  Whether `PipeOp` should selected the supported feature types.
  Otherwise it will err on receiving tasks with unsupported feature
  types.

## Internals

Uses
[`batchgetter_categ()`](https://mlr3torch.mlr-org.com/dev/reference/batchgetter_categ.md).

## Input and Output Channels

One input channel called `"input"` and one output channel called
`"output"`. For an explanation see
[`PipeOpTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch.md).

## State

The state is set to the input shape.

## See also

Other Graph Network:
[`ModelDescriptor()`](https://mlr3torch.mlr-org.com/dev/reference/ModelDescriptor.md),
[`TorchIngressToken()`](https://mlr3torch.mlr-org.com/dev/reference/TorchIngressToken.md),
[`mlr_learners_torch_model`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners_torch_model.md),
[`mlr_pipeops_module`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_module.md),
[`mlr_pipeops_torch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch.md),
[`mlr_pipeops_torch_ingress`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch_ingress.md),
[`mlr_pipeops_torch_ingress_ltnsr`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch_ingress_ltnsr.md),
[`mlr_pipeops_torch_ingress_num`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch_ingress_num.md),
[`model_descriptor_to_learner()`](https://mlr3torch.mlr-org.com/dev/reference/model_descriptor_to_learner.md),
[`model_descriptor_to_module()`](https://mlr3torch.mlr-org.com/dev/reference/model_descriptor_to_module.md),
[`model_descriptor_union()`](https://mlr3torch.mlr-org.com/dev/reference/model_descriptor_union.md),
[`nn_graph()`](https://mlr3torch.mlr-org.com/dev/reference/nn_graph.md)

## Super classes

[`mlr3pipelines::PipeOp`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html)
-\>
[`PipeOpTorchIngress`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch_ingress.md)
-\> `PipeOpTorchIngressCategorical`

## Methods

### Public methods

- [`PipeOpTorchIngressCategorical$new()`](#method-PipeOpTorchIngressCategorical-initialize)

- [`PipeOpTorchIngressCategorical$clone()`](#method-PipeOpTorchIngressCategorical-clone)

Inherited methods

- [`mlr3pipelines::PipeOp$help()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-help)
- [`mlr3pipelines::PipeOp$predict()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-predict)
- [`mlr3pipelines::PipeOp$print()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-print)
- [`mlr3pipelines::PipeOp$train()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-train)

------------------------------------------------------------------------

### `PipeOpTorchIngressCategorical$new()`

Creates a new instance of this
[R6](https://r6.r-lib.org/reference/R6Class.html) class.

#### Usage

    PipeOpTorchIngressCategorical$new(
      id = "torch_ingress_categ",
      param_vals = list()
    )

#### Arguments

- `id`:

  (`character(1)`)  
  Identifier of the resulting object.

- `param_vals`:

  ([`list()`](https://rdrr.io/r/base/list.html))  
  List of hyperparameter settings, overwriting the hyperparameter
  settings that would otherwise be set during construction.

------------------------------------------------------------------------

### `PipeOpTorchIngressCategorical$clone()`

The objects of this class are cloneable with this method.

#### Usage

    PipeOpTorchIngressCategorical$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.

## Examples

``` r
graph = po("select", selector = selector_type("factor")) %>>%
  po("torch_ingress_categ")
task = tsk("german_credit")
# The output is a model descriptor
md = graph$train(task)[[1L]]
ingress = md$ingress[[1L]]
ingress$batchgetter(task$data(1, ingress$features(task)), "cpu")
#> torch_tensor
#>  5  5  1  2  3  1  3  1  3  1  4  5  1  2
#> [ CPULongType{1,14} ]
```
