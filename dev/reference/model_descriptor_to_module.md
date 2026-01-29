# Create a nn_graph from ModelDescriptor

Creates the
[`nn_graph`](https://mlr3torch.mlr-org.com/dev/reference/nn_graph.md)
from a
[`ModelDescriptor`](https://mlr3torch.mlr-org.com/dev/reference/ModelDescriptor.md).
Mostly for internal use, since the
[`ModelDescriptor`](https://mlr3torch.mlr-org.com/dev/reference/ModelDescriptor.md)
is in most circumstances harder to use than just creating
[`nn_graph`](https://mlr3torch.mlr-org.com/dev/reference/nn_graph.md)
directly.

## Usage

``` r
model_descriptor_to_module(
  model_descriptor,
  output_pointers = NULL,
  list_output = FALSE
)
```

## Arguments

- model_descriptor:

  ([`ModelDescriptor`](https://mlr3torch.mlr-org.com/dev/reference/ModelDescriptor.md))  
  Model Descriptor. `pointer` is ignored, instead `output_pointer`
  values are used. `$graph` member is modified by-reference.

- output_pointers:

  (`list` of `character`)  
  Collection of `pointer`s that indicate what part of the
  `model_descriptor$graph` is being used for output. Entries have the
  format of `ModelDescriptor$pointer`.

- list_output:

  (`logical(1)`)  
  Whether output should be a list of tensors. If `FALSE`, then
  `length(output_pointers)` must be 1.

## Value

[`nn_graph`](https://mlr3torch.mlr-org.com/dev/reference/nn_graph.md)

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
[`model_descriptor_to_learner()`](https://mlr3torch.mlr-org.com/dev/reference/model_descriptor_to_learner.md),
[`model_descriptor_union()`](https://mlr3torch.mlr-org.com/dev/reference/model_descriptor_union.md),
[`nn_graph()`](https://mlr3torch.mlr-org.com/dev/reference/nn_graph.md)
