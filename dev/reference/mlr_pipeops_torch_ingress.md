# Entrypoint to Torch Network

Use this as entry-point to mlr3torch-networks. Unless you are an
advanced user, you should not need to use this directly but
[`PipeOpTorchIngressNumeric`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch_ingress_num.md),
[`PipeOpTorchIngressCategorical`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch_ingress_categ.md)
or
[`PipeOpTorchIngressLazyTensor`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch_ingress_ltnsr.md).

## Parameters

Defined by the construction argument `param_set`.

## Input and Output Channels

One input channel called `"input"` and one output channel called
`"output"`. For an explanation see
[`PipeOpTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch.md).

## State

The state is set to the input shape.

## Internals

Creates an object of class
[`TorchIngressToken`](https://mlr3torch.mlr-org.com/dev/reference/TorchIngressToken.md)
for the given task. The purpuse of this is to store the information on
how to construct the torch dataloader from the task for this entry point
of the network.

## See also

Other Graph Network:
[`ModelDescriptor()`](https://mlr3torch.mlr-org.com/dev/reference/ModelDescriptor.md),
[`TorchIngressToken()`](https://mlr3torch.mlr-org.com/dev/reference/TorchIngressToken.md),
[`mlr_learners_torch_model`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners_torch_model.md),
[`mlr_pipeops_module`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_module.md),
[`mlr_pipeops_torch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch.md),
[`mlr_pipeops_torch_ingress_categ`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch_ingress_categ.md),
[`mlr_pipeops_torch_ingress_ltnsr`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch_ingress_ltnsr.md),
[`mlr_pipeops_torch_ingress_num`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch_ingress_num.md),
[`model_descriptor_to_learner()`](https://mlr3torch.mlr-org.com/dev/reference/model_descriptor_to_learner.md),
[`model_descriptor_to_module()`](https://mlr3torch.mlr-org.com/dev/reference/model_descriptor_to_module.md),
[`model_descriptor_union()`](https://mlr3torch.mlr-org.com/dev/reference/model_descriptor_union.md),
[`nn_graph()`](https://mlr3torch.mlr-org.com/dev/reference/nn_graph.md)

## Super class

[`mlr3pipelines::PipeOp`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html)
-\> `PipeOpTorchIngress`

## Active bindings

- `feature_types`:

  (`character(1)`)  
  The features types that can be consumed by this `PipeOpTorchIngress`.

## Methods

### Public methods

- [`PipeOpTorchIngress$new()`](#method-PipeOpTorchIngress-initialize)

- [`PipeOpTorchIngress$clone()`](#method-PipeOpTorchIngress-clone)

Inherited methods

- [`mlr3pipelines::PipeOp$help()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-help)
- [`mlr3pipelines::PipeOp$predict()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-predict)
- [`mlr3pipelines::PipeOp$print()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-print)
- [`mlr3pipelines::PipeOp$train()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-train)

------------------------------------------------------------------------

### `PipeOpTorchIngress$new()`

Creates a new instance of this
[R6](https://r6.r-lib.org/reference/R6Class.html) class.

#### Usage

    PipeOpTorchIngress$new(
      id,
      param_set = ps(),
      param_vals = list(),
      packages = character(0),
      feature_types
    )

#### Arguments

- `id`:

  (`character(1)`)  
  Identifier of the resulting object.

- `param_set`:

  ([`ParamSet`](https://paradox.mlr-org.com/reference/ParamSet.html))  
  The parameter set.

- `param_vals`:

  ([`list()`](https://rdrr.io/r/base/list.html))  
  List of hyperparameter settings, overwriting the hyperparameter
  settings that would otherwise be set during construction.

- `packages`:

  ([`character()`](https://rdrr.io/r/base/character.html))  
  The R packages this object depends on.

- `feature_types`:

  ([`character()`](https://rdrr.io/r/base/character.html))  
  The feature types. See
  [`mlr_reflections$task_feature_types`](https://mlr3.mlr-org.com/reference/mlr_reflections.html)
  for available values, Additionally, `"lazy_tensor"` is supported.

------------------------------------------------------------------------

### `PipeOpTorchIngress$clone()`

The objects of this class are cloneable with this method.

#### Usage

    PipeOpTorchIngress$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.
