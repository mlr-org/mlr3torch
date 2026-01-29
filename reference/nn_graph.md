# Graph Network

Represents a neural network using a
[`Graph`](https://mlr3pipelines.mlr-org.com/reference/Graph.html) that
contains mostly
[`PipeOpModule`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_module.md)s.

## Usage

``` r
nn_graph(graph, shapes_in, output_map = graph$output$name, list_output = FALSE)
```

## Arguments

- graph:

  ([`Graph`](https://mlr3pipelines.mlr-org.com/reference/Graph.html))  
  The [`Graph`](https://mlr3pipelines.mlr-org.com/reference/Graph.html)
  to wrap. Is **not** cloned.

- shapes_in:

  (named `integer`)  
  Shape info of tensors that go into `graph`. Names must be
  `graph$input$name`, possibly in different order.

- output_map:

  (`character`)  
  Which of `graph`'s outputs to use. Must be a subset of
  `graph$output$name`.

- list_output:

  (`logical(1)`)  
  Whether output should be a list of tensors. If `FALSE` (default), then
  `length(output_map)` must be 1.

## Value

`nn_graph`

## Fields

- `graph` ::
  [`Graph`](https://mlr3pipelines.mlr-org.com/reference/Graph.html)  
  The graph (consisting primarily of
  [`PipeOpModule`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_module.md)s)
  that is wrapped by the network.

- `input_map` ::
  [`character()`](https://rdrr.io/r/base/character.html)  
  The names of the input arguments of the network.

- `shapes_in` :: [`list()`](https://rdrr.io/r/base/list.html)  
  The shapes of the input tensors of the network.

- `output_map` ::
  [`character()`](https://rdrr.io/r/base/character.html)  
  Which output elements of the graph are returned by the `$forward()`
  method.

- `list_output` :: `logical(1)`  
  Whether the output is a list of tensors.

- `module_list` ::
  [`nn_module_list`](https://torch.mlverse.org/docs/reference/nn_module_list.html)  
  The list of modules in the network.

- `list_output` :: `logical(1)`  
  Whether the output is a list of tensors.

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
[`model_descriptor_union()`](https://mlr3torch.mlr-org.com/reference/model_descriptor_union.md)

## Examples

``` r
graph = mlr3pipelines::Graph$new()
graph$add_pipeop(po("module_1", module = nn_linear(10, 20)), clone = FALSE)
graph$add_pipeop(po("module_2", module = nn_relu()), clone = FALSE)
graph$add_pipeop(po("module_3", module = nn_linear(20, 1)), clone = FALSE)
graph$add_edge("module_1", "module_2")
graph$add_edge("module_2", "module_3")

network = nn_graph(graph, shapes_in = list(module_1.input = c(NA, 10)))

x = torch_randn(16, 10)

network(module_1.input = x)
#> torch_tensor
#>  0.3073
#>  0.0269
#>  0.0292
#>  0.2388
#> -0.0150
#> -0.0077
#>  0.0522
#>  0.1992
#>  0.0302
#>  0.4510
#>  0.2953
#>  0.2050
#>  0.0754
#>  0.4327
#>  0.4228
#>  0.3021
#> [ CPUFloatType{16,1} ][ grad_fn = <AddmmBackward0> ]
```
