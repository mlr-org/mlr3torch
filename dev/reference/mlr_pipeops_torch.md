# Base Class for Torch Module Constructor Wrappers

`PipeOpTorch` is the base class for all
[`PipeOp`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html)s
that represent neural network layers in a
[`Graph`](https://mlr3pipelines.mlr-org.com/reference/Graph.html). If
you want to create your own `PipeOpTorch`, check out the
[`pipeop_torch()`](https://mlr3torch.mlr-org.com/dev/reference/pipeop_torch.md)
helper function and the article [Writing your own
PipeOpTorch](https://mlr3torch.mlr-org.com/articles/custom_pipeop_torch.html).
During **training**, it generates a
[`PipeOpModule`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_module.md)
that wraps an
[`nn_module`](https://torch.mlverse.org/docs/reference/nn_module.html)
and attaches it to the architecture, which is also represented as a
[`Graph`](https://mlr3pipelines.mlr-org.com/reference/Graph.html)
consisting mostly of
[`PipeOpModule`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_module.md)s
an
[`PipeOpNOP`](https://mlr3pipelines.mlr-org.com/reference/mlr_pipeops_nop.html)s.

The convenient way to construct such a `PipeOp` is the
[`nn()`](https://mlr3torch.mlr-org.com/dev/reference/nn.md) helper,
which prefixes the given key with `"nn_"` to look it up in the
[`mlr_pipeops`](https://mlr3pipelines.mlr-org.com/reference/mlr_pipeops.html)
dictionary and uses the unprefixed key as the id of the resulting
`PipeOp`: `nn("linear", out_features = 10)` is equivalent to
`po("nn_linear", id = "linear", out_features = 10)`. Because ids must be
unique within a
[`Graph`](https://mlr3pipelines.mlr-org.com/reference/Graph.html),
repeated layers can be disambiguated with a `_<n>` suffix, e.g.
`nn("linear_1")` and `nn("linear_2")`.

While the former
[`Graph`](https://mlr3pipelines.mlr-org.com/reference/Graph.html)
operates on
[`ModelDescriptor`](https://mlr3torch.mlr-org.com/dev/reference/ModelDescriptor.md)s,
the latter operates on
[tensors](https://torch.mlverse.org/docs/reference/torch_tensor.html).

The relationship between a `PipeOpTorch` and a
[`PipeOpModule`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_module.md)
is similar to the relationshop between a `nn_module_generator` (like
[`nn_linear`](https://torch.mlverse.org/docs/reference/nn_linear.html))
and a
[`nn_module`](https://torch.mlverse.org/docs/reference/nn_module.html)
(like the output of `nn_linear(...)`). A crucial difference is that the
`PipeOpTorch` infers auxiliary parameters (like `in_features` for
`nn_linear`) automatically from the intermediate tensor shapes that are
being communicated through the
[`ModelDescriptor`](https://mlr3torch.mlr-org.com/dev/reference/ModelDescriptor.md).

During **prediction**, `PipeOpTorch` takes in a
[`Task`](https://mlr3.mlr-org.com/reference/Task.html) in each channel
and outputs the same new
[`Task`](https://mlr3.mlr-org.com/reference/Task.html) resulting from
their [feature
union](https://mlr3pipelines.mlr-org.com/reference/mlr_pipeops_featureunion.html)
in each channel. If there is only one input and output channel, the task
is simply piped through.

## Parameters

The [`ParamSet`](https://paradox.mlr-org.com/reference/ParamSet.html) is
specified by the child class inheriting from `PipeOpTorch`. Usually the
parameters are the arguments of the wrapped
[`nn_module`](https://torch.mlverse.org/docs/reference/nn_module.html)
minus the auxiliary parameter that can be automatically inferred from
the shapes of the input tensors.

## Inheriting

When inheriting from this class, one should overload either the
`private$.shapes_out()` and the `private$.shape_dependent_params()`
methods, or overload `private$.make_module()`.

- `.make_module(shapes_in, param_vals, task)`  
  ([`list()`](https://rdrr.io/r/base/list.html),
  [`list()`](https://rdrr.io/r/base/list.html),
  [`Task`](https://mlr3.mlr-org.com/reference/Task.html) or `NULL`) -\>
  `nn_module`  
  This private method is called to generate the `nn_module` that is
  passed as argument `module` to
  [`PipeOpModule`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_module.md).
  It must be overwritten, when no `module_generator` is provided. If
  left as is, it calls the provided `module_generator` with the
  arguments obtained by the private method `.shape_dependent_params()`.

- `.shapes_out(shapes_in, param_vals, task)`  
  ([`list()`](https://rdrr.io/r/base/list.html),
  [`list()`](https://rdrr.io/r/base/list.html),
  [`Task`](https://mlr3.mlr-org.com/reference/Task.html) or `NULL`) -\>
  named [`list()`](https://rdrr.io/r/base/list.html)  
  This private method gets a list of `integer` vectors (`shapes_in`),
  the parameter values (`param_vals`), as well as an (optional)
  [`Task`](https://mlr3.mlr-org.com/reference/Task.html). The
  `shapes_in` list indicates the shape of input tensors that will be fed
  to the module's `$forward()` function. The list has one item per input
  tensor, typically only one. The function should return a list of
  shapes of tensors that are created by the module. The `shapes_in` are
  named after the input channels of the `PipeOp` and are in the same
  order. The output shapes must be in the same order as the output names
  of the `PipeOp`. In case the output shapes depends on the task (as is
  the case for
  [`PipeOpTorchHead`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_head.md)),
  the function should return valid output shapes (possibly containing
  `NA`s) whether or not the `task` argument is provided. Any dimension
  of `shapes_in` can be `NA`, i.e. unknown, so this method must not
  assume that a dimension it reads is known. It has to assert the
  dimensions it actually needs and propagate the `NA`s it can live with.

  The inference should generally be **permissive**. For example,
  applying a convolutional layer to an input of shape `c(NA, 3, NA, NA)`
  should assume that the last two dimensions are big enough for the
  given kernel size. This can sometimes lead to runtime errors but this
  is preferable over rejecting valid architectures.

  There are various assertion helpers for common input checks, such as
  [`assert_known_dims()`](https://mlr3torch.mlr-org.com/dev/reference/assert_known_dims.md)
  and the "See also" links on its page. There are also
  [`shape_helpers`](https://mlr3torch.mlr-org.com/dev/reference/shape_helpers.md),
  which provide the shape arithmetic (broadcasting, resolving negative
  dimension indices).

- `.shape_dependent_params(shapes_in, param_vals, task)`  
  ([`list()`](https://rdrr.io/r/base/list.html),
  [`list()`](https://rdrr.io/r/base/list.html),
  [`Task`](https://mlr3.mlr-org.com/reference/Task.html) or `NULL`) -\>
  named [`list()`](https://rdrr.io/r/base/list.html)  
  This private method has the same inputs as `.shapes_out`. If
  `.make_module()` is not overwritten, it constructs the arguments
  passed to `module_generator`. Usually this means that it must infer
  the auxiliary parameters that can be inferred from the input shapes
  and add it to the user-supplied parameter values (`param_vals`).

## Input and Output Channels

During *training*, all inputs and outputs are of class
[`ModelDescriptor`](https://mlr3torch.mlr-org.com/dev/reference/ModelDescriptor.md).
During *prediction*, all input and output channels are of class
[`Task`](https://mlr3.mlr-org.com/reference/Task.html).

## Shape Inference

A network is assembled without any data flowing through it, so mlr3torch
tracks the shape of the tensors instead: it starts from the shape the
ingress announces and hands each `PipeOp` the shapes of its inputs,
which is how auxiliary parameters such as `in_features` of
[`nn_linear`](https://torch.mlverse.org/docs/reference/nn_linear.html)
are filled in automatically.

A shape is an [`integer()`](https://rdrr.io/r/base/integer.html) whose
first entry is the batch dimension, e.g. `c(NA, 3, 32, 32)` for a batch
of RGB images of 32 by 32 pixels. A dimension whose size is not known in
advance is `NA`, and any dimension can be `NA`, not only the batch
dimension: the sequence length of a transformer or the height and width
of an image may equally well vary.

The public `$shapes_out()` method answers the same question for a single
`PipeOp`, without building a network around it, which is the quickest
way to see what an operator makes of a given input shape. It is also
what reports the problem when an operator needs a dimension that is
unknown, naming the `PipeOp` and the shape it was given.

Implementing a `PipeOp` that computes its own output shapes is described
under `.shapes_out()` in the "Inheriting" section above.

## State

The state is the value calculated by the public method `shapes_out()`.

## Internals

During training, the `PipeOpTorch` creates a
[`PipeOpModule`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_module.md)
for the given parameter specification and the input shapes from the
incoming
[`ModelDescriptor`](https://mlr3torch.mlr-org.com/dev/reference/ModelDescriptor.md)s
using the private method `.make_module()`. The input shapes are provided
by the slot `pointer_shape` of the incoming
[`ModelDescriptor`](https://mlr3torch.mlr-org.com/dev/reference/ModelDescriptor.md)s.
The channel names of this
[`PipeOpModule`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_module.md)
are identical to the channel names of the generating `PipeOpTorch`.

A [model descriptor
union](https://mlr3torch.mlr-org.com/dev/reference/model_descriptor_union.md)
of all incoming
[`ModelDescriptor`](https://mlr3torch.mlr-org.com/dev/reference/ModelDescriptor.md)s
is then created. Note that this modifies the
[`graph`](https://mlr3pipelines.mlr-org.com/reference/Graph.html) of the
first
[`ModelDescriptor`](https://mlr3torch.mlr-org.com/dev/reference/ModelDescriptor.md)
**in place** for efficiency. The
[`PipeOpModule`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_module.md)
is added to the
[`graph`](https://mlr3pipelines.mlr-org.com/reference/Graph.html) slot
of this union and the edges that connect the sending `PipeOpModule`s to
the input channel of this `PipeOpModule` are addeded to the graph. This
is possible because every incoming
[`ModelDescriptor`](https://mlr3torch.mlr-org.com/dev/reference/ModelDescriptor.md)
contains the information about the `id` and the `channel` name of the
sending `PipeOp` in the slot `pointer`.

The new graph in the
[`model_descriptor_union`](https://mlr3torch.mlr-org.com/dev/reference/model_descriptor_union.md)
represents the current state of the neural network architecture. It is
structurally similar to the subgraph that consists of all pipeops of
class `PipeOpTorch` and
[`PipeOpTorchIngress`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch_ingress.md)
that are ancestors of this `PipeOpTorch`.

For the output, a shallow copy of the
[`ModelDescriptor`](https://mlr3torch.mlr-org.com/dev/reference/ModelDescriptor.md)
is created and the `pointer` and `pointer_shape` are updated
accordingly. The shallow copy means that all
[`ModelDescriptor`](https://mlr3torch.mlr-org.com/dev/reference/ModelDescriptor.md)s
point to the same
[`Graph`](https://mlr3pipelines.mlr-org.com/reference/Graph.html) which
allows the graph to be modified by-reference in different parts of the
code.

## See also

Other Graph Network:
[`ModelDescriptor()`](https://mlr3torch.mlr-org.com/dev/reference/ModelDescriptor.md),
[`TorchIngressToken()`](https://mlr3torch.mlr-org.com/dev/reference/TorchIngressToken.md),
[`mlr_learners_torch_model`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners_torch_model.md),
[`mlr_pipeops_module`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_module.md),
[`mlr_pipeops_torch_ingress`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch_ingress.md),
[`mlr_pipeops_torch_ingress_categ`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch_ingress_categ.md),
[`mlr_pipeops_torch_ingress_ltnsr`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch_ingress_ltnsr.md),
[`mlr_pipeops_torch_ingress_num`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch_ingress_num.md),
[`model_descriptor_to_learner()`](https://mlr3torch.mlr-org.com/dev/reference/model_descriptor_to_learner.md),
[`model_descriptor_to_module()`](https://mlr3torch.mlr-org.com/dev/reference/model_descriptor_to_module.md),
[`model_descriptor_union()`](https://mlr3torch.mlr-org.com/dev/reference/model_descriptor_union.md),
[`nn_graph()`](https://mlr3torch.mlr-org.com/dev/reference/nn_graph.md),
[`pipeop_torch()`](https://mlr3torch.mlr-org.com/dev/reference/pipeop_torch.md)

## Super class

[`mlr3pipelines::PipeOp`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html)
-\> `PipeOpTorch`

## Public fields

- `module_generator`:

  (`nn_module_generator` or `NULL`)  
  The module generator wrapped by this `PipeOpTorch`. If `NULL`, the
  private method `private$.make_module(shapes_in, param_vals)` must be
  overwritten, see section 'Inheriting'. Do not change this after
  construction.

## Methods

### Public methods

- [`PipeOpTorch$new()`](#method-PipeOpTorch-initialize)

- [`PipeOpTorch$shapes_out()`](#method-PipeOpTorch-shapes_out)

- [`PipeOpTorch$clone()`](#method-PipeOpTorch-clone)

Inherited methods

- [`mlr3pipelines::PipeOp$help()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-help)
- [`mlr3pipelines::PipeOp$predict()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-predict)
- [`mlr3pipelines::PipeOp$print()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-print)
- [`mlr3pipelines::PipeOp$train()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-train)

------------------------------------------------------------------------

### `PipeOpTorch$new()`

Creates a new instance of this
[R6](https://r6.r-lib.org/reference/R6Class.html) class.

#### Usage

    PipeOpTorch$new(
      id,
      module_generator,
      param_set = ps(),
      param_vals = list(),
      inname = "input",
      outname = "output",
      packages = "torch",
      tags = NULL
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

- `param_vals`:

  ([`list()`](https://rdrr.io/r/base/list.html))  
  List of hyperparameter settings, overwriting the hyperparameter
  settings that would otherwise be set during construction.

- `inname`:

  ([`character()`](https://rdrr.io/r/base/character.html))  
  The names of the
  [`PipeOp`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html)'s
  input channels, `"input"` by default. These will be the input channels
  of the generated
  [`PipeOpModule`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_module.md).
  The tensors are passed to the wrapped `nn_module` by position, i.e.
  the order of the input channels determines which argument of the
  forward method they end up in. Unless the forward method has the
  argument `...`, naming the input channels after its arguments is
  therefore recommended, as it avoids any ambiguity.

- `outname`:

  ([`character()`](https://rdrr.io/r/base/character.html))  
  The names of the output channels, `"output"` by default. These will be
  the ouput channels of the generated
  [`PipeOpModule`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_module.md)
  and therefore also the names of the list returned by its `$train()`.
  In case there is more than one output channel, the `nn_module` that is
  constructed by this
  [`PipeOp`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html)
  during training must return a
  [`list()`](https://rdrr.io/r/base/list.html) whose elements are in the
  order of the output channels.

- `packages`:

  ([`character()`](https://rdrr.io/r/base/character.html))  
  The R packages this object depends on.

- `tags`:

  ([`character()`](https://rdrr.io/r/base/character.html))  
  The tags of the
  [`PipeOp`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html).
  The tags `"torch"` is always added.

------------------------------------------------------------------------

### `PipeOpTorch$shapes_out()`

Calculates the output shapes for the given input shapes, parameters and
task.

#### Usage

    PipeOpTorch$shapes_out(shapes_in, task = NULL)

#### Arguments

- `shapes_in`:

  ([`list()`](https://rdrr.io/r/base/list.html) of
  [`integer()`](https://rdrr.io/r/base/integer.html))  
  The input shapes, which must be in the same order as the input channel
  names of the `PipeOp`.

- `task`:

  ([`Task`](https://mlr3.mlr-org.com/reference/Task.html) or `NULL`)  
  The task, which is very rarely used (default is `NULL`). An exception
  is
  [`PipeOpTorchHead`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_head.md).

#### Returns

A named [`list()`](https://rdrr.io/r/base/list.html) containing the
output shapes. The names are the names of the output channels of the
`PipeOp`.

------------------------------------------------------------------------

### `PipeOpTorch$clone()`

The objects of this class are cloneable with this method.

#### Usage

    PipeOpTorch$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.

## Examples

``` r
# A `PipeOpTorch` does not compute anything itself: it adds a module to the network that is
# being assembled, and reports the shape of the tensors that module will produce.
task = tsk("iris")

po_linear = po("nn_linear", out_features = 10)

# what the operator makes of an input shape, without building a network around it.
# `NA` is an unknown dimension -- the batch size here -- and any dimension may be unknown.
po_linear$shapes_out(list(c(NA, 4)))
#> $output
#> [1] NA 10
#> 

# a dimension the operator does need is reported, naming the PipeOp and the shape it was given
try(po_linear$shapes_out(list(c(NA, NA))))
#> Error : PipeOp 'nn_linear' requires the last dimension (the number of input features) of the input shape to be known, but got shape (NA,NA).

# the graph of `PipeOpTorch`s describes the network, `model_descriptor_to_module()` builds it.
# `in_features` of the linear layer was inferred from the task, `out_features` of the head from
# the number of classes.
graph = po("torch_ingress_num") %>>% po_linear %>>% po("nn_relu") %>>% po("nn_head")
md = graph$train(task)[[1L]]
network = model_descriptor_to_module(md)
network
#> An `nn_module` containing 83 parameters.
#> 
#> ── Modules ─────────────────────────────────────────────────────────────────────
#> • module_list: <nn_module_list> #83 parameters

x = torch_tensor(as.matrix(task$data(1:2, task$feature_names)))
with_no_grad(network(torch_ingress_num.input = x))
#> torch_tensor
#> -0.1741 -0.5103  0.1239
#> -0.1304 -0.4756  0.1082
#> [ CPUFloatType{2,3} ]


## What happens during training

# A `PipeOpTorch` operates on `ModelDescriptor`s, which we here build by hand: two networks that
# each start from one half of the iris task.
task1 = task$clone()$select(paste0("Sepal.", c("Length", "Width")))
task2 = task$clone()$select(paste0("Petal.", c("Length", "Width")))
ingress = gunion(list(po("torch_ingress_num_1"), po("torch_ingress_num_2")))
mds_in = ingress$train(list(task1, task2), single_input = FALSE)

mds_in[[1L]][c("graph", "task", "ingress", "pointer", "pointer_shape")]
#> $graph
#> 
#> ── Graph with 1 PipeOps: ───────────────────────────────────────────────────────
#>                   ID         State sccssors prdcssors
#>               <char>        <char>   <char>    <char>
#>  torch_ingress_num_1 <<UNTRAINED>>                   
#> 
#> ── Pipeline: <INPUT> -> torch_ingress_num_1 -> <OUTPUT> 
#> 
#> $task
#> 
#> ── <TaskClassif> (150x3): Iris Flowers ─────────────────────────────────────────
#> • Target: Species
#> • Properties: multiclass
#> • Features (2):
#>   • dbl (2): Sepal.Length, Sepal.Width
#> • Target classes: setosa (33%), versicolor (33%), virginica (33%)
#> 
#> $ingress
#> $ingress$torch_ingress_num_1.input
#> Ingress: Task[selector_name(c("Sepal.Length", "Sepal.Width"), assert_present = TRUE)] --> Tensor(NA, 2)
#> 
#> 
#> $pointer
#> [1] "torch_ingress_num_1" "output"             
#> 
#> $pointer_shape
#> [1] NA  2
#> 
mds_in[[2L]][c("graph", "task", "ingress", "pointer", "pointer_shape")]
#> $graph
#> 
#> ── Graph with 1 PipeOps: ───────────────────────────────────────────────────────
#>                   ID         State sccssors prdcssors
#>               <char>        <char>   <char>    <char>
#>  torch_ingress_num_2 <<UNTRAINED>>                   
#> 
#> ── Pipeline: <INPUT> -> torch_ingress_num_2 -> <OUTPUT> 
#> 
#> $task
#> 
#> ── <TaskClassif> (150x3): Iris Flowers ─────────────────────────────────────────
#> • Target: Species
#> • Properties: multiclass
#> • Features (2):
#>   • dbl (2): Petal.Length, Petal.Width
#> • Target classes: setosa (33%), versicolor (33%), virginica (33%)
#> 
#> $ingress
#> $ingress$torch_ingress_num_2.input
#> Ingress: Task[selector_name(c("Petal.Length", "Petal.Width"), assert_present = TRUE)] --> Tensor(NA, 2)
#> 
#> 
#> $pointer
#> [1] "torch_ingress_num_2" "output"             
#> 
#> $pointer_shape
#> [1] NA  2
#> 

# Training a `PipeOpTorch` on them creates the `PipeOpModule` that wraps the constructed
# `nn_module`, adds it to the network and connects it to both ingress operators.
po_merge = nn("merge_cat", innum = 2)
md_out = po_merge$train(list(input1 = mds_in[[1L]], input2 = mds_in[[2L]]))[[1L]]

# Note that, for efficiency, the graph of the first input is modified in-place.
identical(md_out$graph, mds_in[[1L]]$graph)
#> [1] TRUE
md_out$graph$edges
#>                 src_id src_channel    dst_id dst_channel
#>                 <char>      <char>    <char>      <char>
#> 1: torch_ingress_num_1      output merge_cat      input1
#> 2: torch_ingress_num_2      output merge_cat      input2

# The task is the feature union of the incoming tasks and `ingress` collects the ingress tokens
# of all incoming `ModelDescriptor`s.
md_out$task
#> 
#> ── <TaskClassif> (150x5): Iris Flowers ─────────────────────────────────────────
#> • Target: Species
#> • Properties: multiclass
#> • Features (4):
#>   • dbl (4): Petal.Length, Petal.Width, Sepal.Length, Sepal.Width
#> • Target classes: setosa (33%), versicolor (33%), virginica (33%)
md_out$ingress
#> $torch_ingress_num_1.input
#> Ingress: Task[selector_name(c("Sepal.Length", "Sepal.Width"), assert_present = TRUE)] --> Tensor(NA, 2)
#> 
#> $torch_ingress_num_2.input
#> Ingress: Task[selector_name(c("Petal.Length", "Petal.Width"), assert_present = TRUE)] --> Tensor(NA, 2)
#> 

# `pointer` and `pointer_shape` now refer to the output of the new module.
md_out$pointer
#> [1] "merge_cat" "output"   
md_out$pointer_shape
#> [1] NA  4

# During prediction, no network is built: the `PipeOpTorch` receives a `Task` in each channel
# and outputs their feature union.
po_merge$predict(list(input1 = task1, input2 = task2))[[1L]]
#> 
#> ── <TaskClassif> (150x5): Iris Flowers ─────────────────────────────────────────
#> • Target: Species
#> • Properties: multiclass
#> • Features (4):
#>   • dbl (4): Petal.Length, Petal.Width, Sepal.Length, Sepal.Width
#> • Target classes: setosa (33%), versicolor (33%), virginica (33%)

# Writing an operator of your own is described in the article "Writing your own PipeOpTorch" on
# the package website and in the section 'Inheriting' above.
```
