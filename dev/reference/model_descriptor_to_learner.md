# Create a Torch Learner from a ModelDescriptor

First a
[`nn_graph`](https://mlr3torch.mlr-org.com/dev/reference/nn_graph.md) is
created using
[`model_descriptor_to_module`](https://mlr3torch.mlr-org.com/dev/reference/model_descriptor_to_module.md)
and then a learner is created from this module and the remaining
information from the model descriptor, which must include the optimizer
and loss function and optionally callbacks.

## Usage

``` r
model_descriptor_to_learner(model_descriptor)
```

## Arguments

- model_descriptor:

  ([`ModelDescriptor`](https://mlr3torch.mlr-org.com/dev/reference/ModelDescriptor.md))  
  The model descriptor.

## Value

[`Learner`](https://mlr3.mlr-org.com/reference/Learner.html)

## See also

Other Graph Network:
[`ModelDescriptor()`](https://mlr3torch.mlr-org.com/dev/reference/ModelDescriptor.md),
[`TorchIngressToken()`](https://mlr3torch.mlr-org.com/dev/reference/TorchIngressToken.md),
[`mlr_learners_torch_model`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners_torch_model.md),
[`mlr_pipeops_module`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_module.md),
[`mlr_pipeops_torch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch.md),
[`mlr_pipeops_torch_ingress`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch_ingress.md),
[`mlr_pipeops_torch_ingress_categ`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch_ingress_categ.md),
[`mlr_pipeops_torch_ingress_ltnsr`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch_ingress_ltnsr.md),
[`mlr_pipeops_torch_ingress_num`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch_ingress_num.md),
[`model_descriptor_to_module()`](https://mlr3torch.mlr-org.com/dev/reference/model_descriptor_to_module.md),
[`model_descriptor_union()`](https://mlr3torch.mlr-org.com/dev/reference/model_descriptor_union.md),
[`nn_graph()`](https://mlr3torch.mlr-org.com/dev/reference/nn_graph.md)
