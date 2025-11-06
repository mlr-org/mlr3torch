# Union of ModelDescriptors

This is a mostly internal function that is used in
[`PipeOpTorch`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_torch.md)s
with multiple input channels.

It creates the union of multiple
[`ModelDescriptor`](https://mlr3torch.mlr-org.com/reference/ModelDescriptor.md)s:

- `graph`s are combinded (if they are not identical to begin with). The
  first entry's `graph` is modified by reference.

- `PipeOp`s with the same ID must be identical. No new input edges may
  be added to `PipeOp`s.

- Drops `pointer` / `pointer_shape` entries.

- The new task is the [feature
  union](https://mlr3pipelines.mlr-org.com/reference/mlr_pipeops_featureunion.html)
  of the two incoming tasks.

- The `optimizer` and `loss` of both
  [`ModelDescriptor`](https://mlr3torch.mlr-org.com/reference/ModelDescriptor.md)s
  must be identical.

- Ingress tokens and callbacks are merged, where objects with the same
  `"id"` must be identical.

## Usage

``` r
model_descriptor_union(md1, md2)
```

## Arguments

- md1:

  (`ModelDescriptor`) The first
  [`ModelDescriptor`](https://mlr3torch.mlr-org.com/reference/ModelDescriptor.md).

- md2:

  (`ModelDescriptor`) The second
  [`ModelDescriptor`](https://mlr3torch.mlr-org.com/reference/ModelDescriptor.md).

## Value

[`ModelDescriptor`](https://mlr3torch.mlr-org.com/reference/ModelDescriptor.md)

## Details

The requirement that no new input edgedes may be added to `PipeOp`s is
not theoretically necessary, but since we assume that ModelDescriptor is
being built from beginning to end (i.e. `PipeOp`s never get new
ancestors) we can make this assumption and simplify things. Otherwise
we'd need to treat "..."-inputs special.)

## See also

Other Graph Network:
[`ModelDescriptor()`](https://mlr3torch.mlr-org.com/reference/ModelDescriptor.md),
[`TorchIngressToken()`](https://mlr3torch.mlr-org.com/reference/TorchIngressToken.md),
[`mlr_learners_torch_model`](https://mlr3torch.mlr-org.com/reference/mlr_learners_torch_model.md),
[`mlr_pipeops_module`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_module.md),
[`mlr_pipeops_torch`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_torch.md),
[`mlr_pipeops_torch_ingress`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_torch_ingress.md),
[`mlr_pipeops_torch_ingress_categ`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_torch_ingress_categ.md),
[`mlr_pipeops_torch_ingress_ltnsr`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_torch_ingress_ltnsr.md),
[`mlr_pipeops_torch_ingress_num`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_torch_ingress_num.md),
[`model_descriptor_to_learner()`](https://mlr3torch.mlr-org.com/reference/model_descriptor_to_learner.md),
[`model_descriptor_to_module()`](https://mlr3torch.mlr-org.com/reference/model_descriptor_to_module.md),
[`nn_graph()`](https://mlr3torch.mlr-org.com/reference/nn_graph.md)

Other Model Configuration:
[`ModelDescriptor()`](https://mlr3torch.mlr-org.com/reference/ModelDescriptor.md),
[`mlr_pipeops_torch_callbacks`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_torch_callbacks.md),
[`mlr_pipeops_torch_loss`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_torch_loss.md),
[`mlr_pipeops_torch_optimizer`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_torch_optimizer.md)
