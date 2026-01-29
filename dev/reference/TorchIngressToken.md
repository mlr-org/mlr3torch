# Torch Ingress Token

This function creates an S3 class of class `"TorchIngressToken"`, which
is an internal data structure. It contains the (meta-)information of how
a batch is generated from a
[`Task`](https://mlr3.mlr-org.com/reference/Task.html) and fed into an
entry point of the neural network. It is stored as the `ingress` field
in a
[`ModelDescriptor`](https://mlr3torch.mlr-org.com/dev/reference/ModelDescriptor.md).

## Usage

``` r
TorchIngressToken(features, batchgetter, shape = NULL)
```

## Arguments

- features:

  (`character` or
  [`mlr3pipelines::Selector`](https://mlr3pipelines.mlr-org.com/reference/Selector.html))  
  Features on which the batchgetter will operate or a selector (such as
  [`mlr3pipelines::selector_type`](https://mlr3pipelines.mlr-org.com/reference/Selector.html)).

- batchgetter:

  (`function`)  
  Function with two arguments: `data` and `device`. This function is
  given the output of `Task$data(rows = batch_indices, cols = features)`
  and it should produce a tensor of shape `shape_out`.

- shape:

  (`integer`)  
  Shape that `batchgetter` will produce. Batch dimension must be
  included as `NA` (but other dimensions can also be `NA`, i.e.,
  unknown).

## Value

`TorchIngressToken` object.

## See also

Other Graph Network:
[`ModelDescriptor()`](https://mlr3torch.mlr-org.com/dev/reference/ModelDescriptor.md),
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

## Examples

``` r
# Define a task for which we want to define an ingress token
task = tsk("iris")

# We create an ingress token for two feature Sepal.Length and Petal.Length:
# We have to specify the features, the batchgetter and the shape
features = c("Sepal.Length", "Petal.Length")
# As a batchgetter we use batchgetter_num

batch_dt = task$data(rows = 1:10, cols =features)
batch_dt
#>     Sepal.Length Petal.Length
#>            <num>        <num>
#>  1:          5.1          1.4
#>  2:          4.9          1.4
#>  3:          4.7          1.3
#>  4:          4.6          1.5
#>  5:          5.0          1.4
#>  6:          5.4          1.7
#>  7:          4.6          1.4
#>  8:          5.0          1.5
#>  9:          4.4          1.4
#> 10:          4.9          1.5
batch_tensor = batchgetter_num(batch_dt, "cpu")
batch_tensor
#> torch_tensor
#>  5.1000  1.4000
#>  4.9000  1.4000
#>  4.7000  1.3000
#>  4.6000  1.5000
#>  5.0000  1.4000
#>  5.4000  1.7000
#>  4.6000  1.4000
#>  5.0000  1.5000
#>  4.4000  1.4000
#>  4.9000  1.5000
#> [ CPUFloatType{10,2} ]

# The shape is unknown in the first dimension (batch dimension)

ingress_token = TorchIngressToken(
  features = features,
  batchgetter = batchgetter_num,
  shape = c(NA, 2)
)
ingress_token
#> Ingress: Task[selector_name(c("Sepal.Length", "Petal.Length"), assert_present = TRUE)] --> Tensor(NA, 2)
```
