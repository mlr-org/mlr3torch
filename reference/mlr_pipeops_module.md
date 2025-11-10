# Class for Torch Module Wrappers

`PipeOpModule` wraps an
[`nn_module`](https://torch.mlverse.org/docs/reference/nn_module.html)
or `function` that is being called during the `train` phase of this
[`mlr3pipelines::PipeOp`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html).
By doing so, this allows to assemble `PipeOpModule`s in a computational
[`mlr3pipelines::Graph`](https://mlr3pipelines.mlr-org.com/reference/Graph.html)
that represents either a neural network or a preprocessing graph of a
[`lazy_tensor`](https://mlr3torch.mlr-org.com/reference/lazy_tensor.md).
In most cases it is easier to create such a network by creating a graph
that generates this graph.

In most cases it is easier to create such a network by creating a
structurally related graph consisting of nodes of class
[`PipeOpTorchIngress`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_torch_ingress.md)
and
[`PipeOpTorch`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_torch.md).
This graph will then generate the graph consisting of `PipeOpModule`s as
part of the
[`ModelDescriptor`](https://mlr3torch.mlr-org.com/reference/ModelDescriptor.md).

## Input and Output Channels

The number and names of the input and output channels can be set during
construction. They input and output `"torch_tensor"` during training,
and `NULL` during prediction as the prediction phase currently serves no
meaningful purpose.

## State

The state is the value calculated by the public method `shapes_out()`.

## Parameters

No parameters.

## Internals

During training, the wrapped
[`nn_module`](https://torch.mlverse.org/docs/reference/nn_module.html) /
`function` is called with the provided inputs in the order in which the
channels are defined. Arguments are **not** matched by name.

## See also

Other Graph Network:
[`ModelDescriptor()`](https://mlr3torch.mlr-org.com/reference/ModelDescriptor.md),
[`TorchIngressToken()`](https://mlr3torch.mlr-org.com/reference/TorchIngressToken.md),
[`mlr_learners_torch_model`](https://mlr3torch.mlr-org.com/reference/mlr_learners_torch_model.md),
[`mlr_pipeops_torch`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_torch.md),
[`mlr_pipeops_torch_ingress`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_torch_ingress.md),
[`mlr_pipeops_torch_ingress_categ`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_torch_ingress_categ.md),
[`mlr_pipeops_torch_ingress_ltnsr`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_torch_ingress_ltnsr.md),
[`mlr_pipeops_torch_ingress_num`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_torch_ingress_num.md),
[`model_descriptor_to_learner()`](https://mlr3torch.mlr-org.com/reference/model_descriptor_to_learner.md),
[`model_descriptor_to_module()`](https://mlr3torch.mlr-org.com/reference/model_descriptor_to_module.md),
[`model_descriptor_union()`](https://mlr3torch.mlr-org.com/reference/model_descriptor_union.md),
[`nn_graph()`](https://mlr3torch.mlr-org.com/reference/nn_graph.md)

Other PipeOp:
[`mlr_pipeops_torch_callbacks`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_torch_callbacks.md),
[`mlr_pipeops_torch_optimizer`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_torch_optimizer.md)

## Super class

[`mlr3pipelines::PipeOp`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html)
-\> `PipeOpModule`

## Public fields

- `module`:

  ([`nn_module`](https://torch.mlverse.org/docs/reference/nn_module.html))  
  The torch module that is called during the training phase.

## Methods

### Public methods

- [`PipeOpModule$new()`](#method-PipeOpModule-new)

- [`PipeOpModule$clone()`](#method-PipeOpModule-clone)

Inherited methods

- [`mlr3pipelines::PipeOp$help()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-help)
- [`mlr3pipelines::PipeOp$predict()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-predict)
- [`mlr3pipelines::PipeOp$print()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-print)
- [`mlr3pipelines::PipeOp$train()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-train)

------------------------------------------------------------------------

### Method `new()`

Creates a new instance of this
[R6](https://r6.r-lib.org/reference/R6Class.html) class.

#### Usage

    PipeOpModule$new(
      id = "module",
      module = nn_identity(),
      inname = "input",
      outname = "output",
      param_vals = list(),
      packages = character(0)
    )

#### Arguments

- `id`:

  (`character(1)`)  
  The id for of the new object.

- `module`:

  ([`nn_module`](https://torch.mlverse.org/docs/reference/nn_module.html)
  or `function()`)  
  The torch module or function that is being wrapped.

- `inname`:

  ([`character()`](https://rdrr.io/r/base/character.html))  
  The names of the input channels.

- `outname`:

  ([`character()`](https://rdrr.io/r/base/character.html))  
  The names of the output channels. If this parameter has length 1, the
  parameter
  [module](https://torch.mlverse.org/docs/reference/nn_module.html) must
  return a
  [tensor](https://torch.mlverse.org/docs/reference/torch_tensor.html).
  Otherwise it must return a
  [`list()`](https://rdrr.io/r/base/list.html) of tensors of
  corresponding length.

- `param_vals`:

  (named [`list()`](https://rdrr.io/r/base/list.html))  
  Parameter values to be set after construction.

- `packages`:

  ([`character()`](https://rdrr.io/r/base/character.html))  
  The R packages this object depends on.

------------------------------------------------------------------------

### Method `clone()`

The objects of this class are cloneable with this method.

#### Usage

    PipeOpModule$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.

## Examples

``` r
## creating an PipeOpModule manually

# one input and output channel
po_module = po("module",
  id = "linear",
  module = torch::nn_linear(10, 20),
  inname = "input",
  outname = "output"
)
x = torch::torch_randn(16, 10)
# This calls the forward function of the wrapped module.
y = po_module$train(list(input = x))
str(y)
#> List of 1
#>  $ output:Float [1:16, 1:20]

# multiple input and output channels
nn_custom = torch::nn_module("nn_custom",
  initialize = function(in_features, out_features) {
    self$lin1 = torch::nn_linear(in_features, out_features)
    self$lin2 = torch::nn_linear(in_features, out_features)
  },
  forward = function(x, z) {
    list(out1 = self$lin1(x), out2 = torch::nnf_relu(self$lin2(z)))
  }
)

module = nn_custom(3, 2)
po_module = po("module",
  id = "custom",
  module = module,
  inname = c("x", "z"),
  outname = c("out1", "out2")
)
x = torch::torch_randn(1, 3)
z = torch::torch_randn(1, 3)
out = po_module$train(list(x = x, z = z))
str(out)
#> List of 2
#>  $ out1:Float [1:1, 1:2]
#>  $ out2:Float [1:1, 1:2]

# How such a PipeOpModule is usually generated
graph = po("torch_ingress_num") %>>% po("nn_linear", out_features = 10L)
result = graph$train(tsk("iris"))
# The PipeOpTorchLinear generates a PipeOpModule and adds it to a new (module) graph
result[[1]]$graph
#> 
#> ── Graph with 2 PipeOps: ───────────────────────────────────────────────────────
#>                 ID         State  sccssors         prdcssors
#>             <char>        <char>    <char>            <char>
#>  torch_ingress_num <<UNTRAINED>> nn_linear                  
#>          nn_linear <<UNTRAINED>>           torch_ingress_num
#> 
#> ── Pipeline: <INPUT> -> torch_ingress_num -> nn_linear -> <OUTPUT> 
linear_module = result[[1L]]$graph$pipeops$nn_linear
linear_module
#> 
#> ── PipeOp <nn_linear>: not trained ─────────────────────────────────────────────
#> Values: list()
#> 
#> ── Input channels: 
#>    name        train predict
#>  <char>       <char>  <char>
#>   input torch_tensor    NULL
#> 
#> ── Output channels: 
#>    name        train predict
#>  <char>       <char>  <char>
#>  output torch_tensor    NULL
formalArgs(linear_module$module)
#> [1] "input"
linear_module$input$name
#> [1] "input"

# Constructing a PipeOpModule using a simple function
po_add1 = po("module",
  id = "add_one",
  module = function(x) x + 1
)
input = list(torch_tensor(1))
po_add1$train(input)$output
#> torch_tensor
#>  2
#> [ CPUFloatType{1} ]
```
