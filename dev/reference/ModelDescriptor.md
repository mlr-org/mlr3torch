# Represent a Model with Meta-Info

Represents a *model*; possibly a complete model, possibly one in the
process of being built up.

This model takes input tensors of shapes `shapes_in` and pipes them
through `graph`. Input shapes get mapped to input channels of `graph`.
Output shapes are named by the output channels of `graph`; it is also
possible to represent no-ops on tensors, in which case names of input
and output should be identical.

`ModelDescriptor` objects typically represent partial models being built
up, in which case the `pointer` slot indicates a specific point in the
graph that produces a tensor of shape `pointer_shape`, on which the
graph should be extended. It is allowed for the `graph` in this
structure to be modified by-reference in different parts of the code.
However, these modifications may never add edges with elements of the
`Graph` as destination. In particular, no element of `graph$input` may
be removed by reference, e.g. by adding an edge to the `Graph` that has
the input channel of a `PipeOp` that was previously without parent as
its destination.

In most cases it is better to create a specific `ModelDescriptor` by
training a
[`Graph`](https://mlr3pipelines.mlr-org.com/reference/Graph.html)
consisting (mostly) of operators
[`PipeOpTorchIngress`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch_ingress.md),
[`PipeOpTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch.md),
[`PipeOpTorchLoss`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch_loss.md),
[`PipeOpTorchOptimizer`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch_optimizer.md),
and
[`PipeOpTorchCallbacks`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch_callbacks.md).

A `ModelDescriptor` can be converted to a
[`nn_graph`](https://mlr3torch.mlr-org.com/dev/reference/nn_graph.md)
via
[`model_descriptor_to_module`](https://mlr3torch.mlr-org.com/dev/reference/model_descriptor_to_module.md).

## Usage

``` r
ModelDescriptor(
  graph,
  ingress,
  task,
  optimizer = NULL,
  loss = NULL,
  callbacks = NULL,
  pointer = NULL,
  pointer_shape = NULL
)
```

## Arguments

- graph:

  ([`Graph`](https://mlr3pipelines.mlr-org.com/reference/Graph.html))  
  `Graph` of
  [`PipeOpModule`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_module.md)
  and
  [`PipeOpNOP`](https://mlr3pipelines.mlr-org.com/reference/mlr_pipeops_nop.html)
  operators.

- ingress:

  (uniquely named `list` of `TorchIngressToken`)  
  List of inputs that go into `graph`. Names of this must be a subset of
  `graph$input$name`.

- task:

  ([`Task`](https://mlr3.mlr-org.com/reference/Task.html))  
  (Training)-Task for which the model is being built. May be necessary
  for for some aspects of what loss to use etc.

- optimizer:

  ([`TorchOptimizer`](https://mlr3torch.mlr-org.com/dev/reference/TorchOptimizer.md)
  \| `NULL`)  
  Additional info: what optimizer to use.

- loss:

  ([`TorchLoss`](https://mlr3torch.mlr-org.com/dev/reference/TorchLoss.md)
  \| `NULL`)  
  Additional info: what loss to use.

- callbacks:

  (A `list` of
  [`CallbackSet`](https://mlr3torch.mlr-org.com/dev/reference/mlr_callback_set.md)
  or `NULL`)  
  Additional info: what callbacks to use.

- pointer:

  (`character(2)` \| `NULL`)  
  Indicating an element on which a model is. Points to an output channel
  within `graph`: Element 1 is the `PipeOp`'s id and element 2 is that
  `PipeOp`'s output channel.

- pointer_shape:

  (`integer` \| `NULL`)  
  Shape of the output indicated by `pointer`.

## Value

(`ModelDescriptor`)

## See also

Other Model Configuration:
[`mlr_pipeops_torch_callbacks`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch_callbacks.md),
[`mlr_pipeops_torch_loss`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch_loss.md),
[`mlr_pipeops_torch_optimizer`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch_optimizer.md),
[`model_descriptor_union()`](https://mlr3torch.mlr-org.com/dev/reference/model_descriptor_union.md)

Other Graph Network:
[`TorchIngressToken()`](https://mlr3torch.mlr-org.com/dev/reference/TorchIngressToken.md),
[`mlr_learners_torch_model`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners_torch_model.md),
[`mlr_pipeops_module`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_module.md),
[`mlr_pipeops_torch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch.md),
[`mlr_pipeops_torch_ingress`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch_ingress.md),
[`mlr_pipeops_torch_ingress_categ`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch_ingress_categ.md),
[`mlr_pipeops_torch_ingress_ltnsr`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch_ingress_ltnsr.md),
[`mlr_pipeops_torch_ingress_num`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch_ingress_num.md),
[`model_descriptor_to_learner()`](https://mlr3torch.mlr-org.com/dev/reference/model_descriptor_to_learner.md),
[`model_descriptor_to_module()`](https://mlr3torch.mlr-org.com/dev/reference/model_descriptor_to_module.md),
[`model_descriptor_union()`](https://mlr3torch.mlr-org.com/dev/reference/model_descriptor_union.md),
[`nn_graph()`](https://mlr3torch.mlr-org.com/dev/reference/nn_graph.md)
