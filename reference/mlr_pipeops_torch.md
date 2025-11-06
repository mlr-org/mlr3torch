# Base Class for Torch Module Constructor Wrappers

`PipeOpTorch` is the base class for all
[`PipeOp`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html)s
that represent neural network layers in a
[`Graph`](https://mlr3pipelines.mlr-org.com/reference/Graph.html).
During **training**, it generates a
[`PipeOpModule`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_module.md)
that wraps an
[`nn_module`](https://torch.mlverse.org/docs/reference/nn_module.html)
and attaches it to the architecture, which is also represented as a
[`Graph`](https://mlr3pipelines.mlr-org.com/reference/Graph.html)
consisting mostly of
[`PipeOpModule`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_module.md)s
an
[`PipeOpNOP`](https://mlr3pipelines.mlr-org.com/reference/mlr_pipeops_nop.html)s.

While the former
[`Graph`](https://mlr3pipelines.mlr-org.com/reference/Graph.html)
operates on
[`ModelDescriptor`](https://mlr3torch.mlr-org.com/reference/ModelDescriptor.md)s,
the latter operates on
[tensors](https://torch.mlverse.org/docs/reference/torch_tensor.html).

The relationship between a `PipeOpTorch` and a
[`PipeOpModule`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_module.md)
is similar to the relationshop between a `nn_module_generator` (like
[`nn_linear`](https://torch.mlverse.org/docs/reference/nn_linear.html))
and a
[`nn_module`](https://torch.mlverse.org/docs/reference/nn_module.html)
(like the output of `nn_linear(...)`). A crucial difference is that the
`PipeOpTorch` infers auxiliary parameters (like `in_features` for
`nn_linear`) automatically from the intermediate tensor shapes that are
being communicated through the
[`ModelDescriptor`](https://mlr3torch.mlr-org.com/reference/ModelDescriptor.md).

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
  [`list()`](https://rdrr.io/r/base/list.html)) -\> `nn_module`  
  This private method is called to generate the `nn_module` that is
  passed as argument `module` to
  [`PipeOpModule`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_module.md).
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
  `shapes_in` can be assumed to be in the same order as the input names
  of the `PipeOp`. The output shapes must be in the same order as the
  output names of the `PipeOp`. In case the output shapes depends on the
  task (as is the case for
  [`PipeOpTorchHead`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_head.md)),
  the function should return valid output shapes (possibly containing
  `NA`s) if the `task` argument is provided or not. It is important to
  properly handle the presence of `NA`s in the input shapes. By default
  (if construction argument `only_batch_unknown` is `TRUE`), only the
  batch dimension can be `NA`. If you set this to `FALSE`, you need to
  take other unknown dimensions into account. The method can also throw
  an error if the input shapes violate some assumptions.

- `.shape_dependent_params(shapes_in, param_vals, task)`  
  ([`list()`](https://rdrr.io/r/base/list.html),
  [`list()`](https://rdrr.io/r/base/list.html)) -\> named
  [`list()`](https://rdrr.io/r/base/list.html)  
  This private method has the same inputs as `.shapes_out`. If
  `.make_module()` is not overwritten, it constructs the arguments
  passed to `module_generator`. Usually this means that it must infer
  the auxiliary parameters that can be inferred from the input shapes
  and add it to the user-supplied parameter values (`param_vals`).

## Input and Output Channels

During *training*, all inputs and outputs are of class
[`ModelDescriptor`](https://mlr3torch.mlr-org.com/reference/ModelDescriptor.md).
During *prediction*, all input and output channels are of class
[`Task`](https://mlr3.mlr-org.com/reference/Task.html).

## State

The state is the value calculated by the public method `shapes_out()`.

## Internals

During training, the `PipeOpTorch` creates a
[`PipeOpModule`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_module.md)
for the given parameter specification and the input shapes from the
incoming
[`ModelDescriptor`](https://mlr3torch.mlr-org.com/reference/ModelDescriptor.md)s
using the private method `.make_module()`. The input shapes are provided
by the slot `pointer_shape` of the incoming
[`ModelDescriptor`](https://mlr3torch.mlr-org.com/reference/ModelDescriptor.md)s.
The channel names of this
[`PipeOpModule`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_module.md)
are identical to the channel names of the generating `PipeOpTorch`.

A [model descriptor
union](https://mlr3torch.mlr-org.com/reference/model_descriptor_union.md)
of all incoming
[`ModelDescriptor`](https://mlr3torch.mlr-org.com/reference/ModelDescriptor.md)s
is then created. Note that this modifies the
[`graph`](https://mlr3pipelines.mlr-org.com/reference/Graph.html) of the
first
[`ModelDescriptor`](https://mlr3torch.mlr-org.com/reference/ModelDescriptor.md)
**in place** for efficiency. The
[`PipeOpModule`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_module.md)
is added to the
[`graph`](https://mlr3pipelines.mlr-org.com/reference/Graph.html) slot
of this union and the the edges that connect the sending `PipeOpModule`s
to the input channel of this `PipeOpModule` are addeded to the graph.
This is possible because every incoming
[`ModelDescriptor`](https://mlr3torch.mlr-org.com/reference/ModelDescriptor.md)
contains the information about the `id` and the `channel` name of the
sending `PipeOp` in the slot `pointer`.

The new graph in the
[`model_descriptor_union`](https://mlr3torch.mlr-org.com/reference/model_descriptor_union.md)
represents the current state of the neural network architecture. It is
structurally similar to the subgraph that consists of all pipeops of
class `PipeOpTorch` and
[`PipeOpTorchIngress`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_torch_ingress.md)
that are ancestors of this `PipeOpTorch`.

For the output, a shallow copy of the
[`ModelDescriptor`](https://mlr3torch.mlr-org.com/reference/ModelDescriptor.md)
is created and the `pointer` and `pointer_shape` are updated
accordingly. The shallow copy means that all
[`ModelDescriptor`](https://mlr3torch.mlr-org.com/reference/ModelDescriptor.md)s
point to the same
[`Graph`](https://mlr3pipelines.mlr-org.com/reference/Graph.html) which
allows the graph to be modified by-reference in different parts of the
code.

## See also

Other Graph Network:
[`ModelDescriptor()`](https://mlr3torch.mlr-org.com/reference/ModelDescriptor.md),
[`TorchIngressToken()`](https://mlr3torch.mlr-org.com/reference/TorchIngressToken.md),
[`mlr_learners_torch_model`](https://mlr3torch.mlr-org.com/reference/mlr_learners_torch_model.md),
[`mlr_pipeops_module`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_module.md),
[`mlr_pipeops_torch_ingress`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_torch_ingress.md),
[`mlr_pipeops_torch_ingress_categ`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_torch_ingress_categ.md),
[`mlr_pipeops_torch_ingress_ltnsr`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_torch_ingress_ltnsr.md),
[`mlr_pipeops_torch_ingress_num`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_torch_ingress_num.md),
[`model_descriptor_to_learner()`](https://mlr3torch.mlr-org.com/reference/model_descriptor_to_learner.md),
[`model_descriptor_to_module()`](https://mlr3torch.mlr-org.com/reference/model_descriptor_to_module.md),
[`model_descriptor_union()`](https://mlr3torch.mlr-org.com/reference/model_descriptor_union.md),
[`nn_graph()`](https://mlr3torch.mlr-org.com/reference/nn_graph.md)

## Super class

[`mlr3pipelines::PipeOp`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html)
-\> `PipeOpTorch`

## Public fields

- `module_generator`:

  (`nn_module_generator` or `NULL`)  
  The module generator wrapped by this `PipeOpTorch`. If `NULL`, the
  private method `private$.make_module(shapes_in, param_vals)` must be
  overwritte, see section 'Inheriting'. Do not change this after
  construction.

## Methods

### Public methods

- [`PipeOpTorch$new()`](#method-PipeOpTorch-new)

- [`PipeOpTorch$shapes_out()`](#method-PipeOpTorch-shapes_out)

- [`PipeOpTorch$clone()`](#method-PipeOpTorch-clone)

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

    PipeOpTorch$new(
      id,
      module_generator,
      param_set = ps(),
      param_vals = list(),
      inname = "input",
      outname = "output",
      packages = "torch",
      tags = NULL,
      only_batch_unknown = TRUE
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
  input channels. These will be the input channels of the generated
  [`PipeOpModule`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_module.md).
  Unless the wrapped `module_generator`'s forward method (if present)
  has the argument `...`, `inname` must be identical to those argument
  names in order to avoid any ambiguity.  
  If the forward method has the argument `...`, the order of the input
  channels determines how the tensors will be passed to the wrapped
  `nn_module`.  
  If left as `NULL` (default), the argument `module_generator` must be
  given and the argument names of the `modue_generator`'s forward
  function are set as `inname`.

- `outname`:

  ([`character()`](https://rdrr.io/r/base/character.html))  
  The names of the output channels channels. These will be the ouput
  channels of the generated
  [`PipeOpModule`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_module.md)
  and therefore also the names of the list returned by its `$train()`.
  In case there is more than one output channel, the `nn_module` that is
  constructed by this
  [`PipeOp`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html)
  during training must return a named
  [`list()`](https://rdrr.io/r/base/list.html), where the names of the
  list are the names out the output channels. The default is `"output"`.

- `packages`:

  ([`character()`](https://rdrr.io/r/base/character.html))  
  The R packages this object depends on.

- `tags`:

  ([`character()`](https://rdrr.io/r/base/character.html))  
  The tags of the
  [`PipeOp`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html).
  The tags `"torch"` is always added.

- `only_batch_unknown`:

  (`logical(1)`)  
  Whether only the batch dimension can be missing in the input shapes or
  whether other dimensions can also be unknown. Default is `TRUE`.

------------------------------------------------------------------------

### Method `shapes_out()`

Calculates the output shapes for the given input shapes, parameters and
task.

#### Usage

    PipeOpTorch$shapes_out(shapes_in, task = NULL)

#### Arguments

- `shapes_in`:

  ([`list()`](https://rdrr.io/r/base/list.html) of
  [`integer()`](https://rdrr.io/r/base/integer.html))  
  The input input shapes, which must be in the same order as the input
  channel names of the `PipeOp`.

- `task`:

  ([`Task`](https://mlr3.mlr-org.com/reference/Task.html) or `NULL`)  
  The task, which is very rarely used (default is `NULL`). An exception
  is
  [`PipeOpTorchHead`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_head.md).

#### Returns

A named [`list()`](https://rdrr.io/r/base/list.html) containing the
output shapes. The names are the names of the output channels of the
`PipeOp`.

------------------------------------------------------------------------

### Method `clone()`

The objects of this class are cloneable with this method.

#### Usage

    PipeOpTorch$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.

## Examples

``` r
## Creating a neural network
# In torch

task = tsk("iris")

network_generator = torch::nn_module(
  initialize = function(task, d_hidden) {
    d_in = length(task$feature_names)
    self$linear = torch::nn_linear(d_in, d_hidden)
    self$output = if (task$task_type == "regr") {
      torch::nn_linear(d_hidden, 1)
    } else if (task$task_type == "classif") {
      torch::nn_linear(d_hidden, output_dim_for(task))
    }
  },
  forward = function(x) {
    x = self$linear(x)
    x = torch::nnf_relu(x)
    self$output(x)
  }
)

network = network_generator(task, d_hidden = 50)
x = torch::torch_tensor(as.matrix(task$data(1, task$feature_names)))
y = torch::with_no_grad(network(x))


# In mlr3torch
network_generator = po("torch_ingress_num") %>>%
  po("nn_linear", out_features = 50) %>>%
  po("nn_head")
md = network_generator$train(task)[[1L]]
network = model_descriptor_to_module(md)
y = torch::with_no_grad(network(torch_ingress_num.input = x))



## Implementing a custom PipeOpTorch

# defining a custom module
nn_custom = nn_module("nn_custom",
  initialize = function(d_in1, d_in2, d_out1, d_out2, bias = TRUE) {
    self$linear1 = nn_linear(d_in1, d_out1, bias)
    self$linear2 = nn_linear(d_in2, d_out2, bias)
  },
  forward = function(input1, input2) {
    output1 = self$linear1(input1)
    output2 = self$linear1(input2)

    list(output1 = output1, output2 = output2)
  }
)

# wrapping the module into a custom PipeOpTorch

library(paradox)

PipeOpTorchCustom = R6::R6Class("PipeOpTorchCustom",
  inherit = PipeOpTorch,
  public = list(
    initialize = function(id = "nn_custom", param_vals = list()) {
      param_set = ps(
        d_out1 = p_int(lower = 1, tags = c("required", "train")),
        d_out2 = p_int(lower = 1, tags = c("required", "train")),
        bias = p_lgl(default = TRUE, tags = "train")
      )
      super$initialize(
        id = id,
        param_vals = param_vals,
        param_set = param_set,
        inname = c("input1", "input2"),
        outname = c("output1", "output2"),
        module_generator = nn_custom
      )
    }
  ),
  private = list(
    .shape_dependent_params = function(shapes_in, param_vals, task) {
      c(param_vals,
        list(d_in1 = tail(shapes_in[["input1"]], 1)), d_in2 = tail(shapes_in[["input2"]], 1)
      )
    },
    .shapes_out = function(shapes_in, param_vals, task) {
      list(
        input1 = c(head(shapes_in[["input1"]], -1), param_vals$d_out1),
        input2 = c(head(shapes_in[["input2"]], -1), param_vals$d_out2)
      )
    }
  )
)

## Training

# generate input
task = tsk("iris")
task1 = task$clone()$select(paste0("Sepal.", c("Length", "Width")))
task2 = task$clone()$select(paste0("Petal.", c("Length", "Width")))
graph = gunion(list(po("torch_ingress_num_1"), po("torch_ingress_num_2")))
mds_in = graph$train(list(task1, task2), single_input = FALSE)

mds_in[[1L]][c("graph", "task", "ingress", "pointer", "pointer_shape")]
#> $graph
#> Graph with 1 PipeOps:
#>                   ID         State sccssors prdcssors
#>               <char>        <char>   <char>    <char>
#>  torch_ingress_num_1 <<UNTRAINED>>                   
#> 
#> $task
#> 
#> ── <TaskClassif> (150x3): Iris Flowers ─────────────────────────────────────────
#> • Target: Species
#> • Target classes: setosa (33%), versicolor (33%), virginica (33%)
#> • Properties: multiclass
#> • Features (2):
#>   • dbl (2): Sepal.Length, Sepal.Width
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
#> Graph with 1 PipeOps:
#>                   ID         State sccssors prdcssors
#>               <char>        <char>   <char>    <char>
#>  torch_ingress_num_2 <<UNTRAINED>>                   
#> 
#> $task
#> 
#> ── <TaskClassif> (150x3): Iris Flowers ─────────────────────────────────────────
#> • Target: Species
#> • Target classes: setosa (33%), versicolor (33%), virginica (33%)
#> • Properties: multiclass
#> • Features (2):
#>   • dbl (2): Petal.Length, Petal.Width
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

# creating the PipeOpTorch and training it
po_torch = PipeOpTorchCustom$new()
po_torch$param_set$values = list(d_out1 = 10, d_out2 = 20)
train_input = list(input1 = mds_in[[1L]], input2 = mds_in[[2L]])
mds_out = do.call(po_torch$train, args = list(input = train_input))
po_torch$state
#> $output1
#> [1] NA 10
#> 
#> $output2
#> [1] NA 20
#> 

# the new model descriptors

# the resulting graphs are identical
identical(mds_out[[1L]]$graph, mds_out[[2L]]$graph)
#> [1] TRUE
# note that as a side-effect, also one of the input graphs is modified in-place for efficiency
mds_in[[1L]]$graph$edges
#>                 src_id src_channel    dst_id dst_channel
#>                 <char>      <char>    <char>      <char>
#> 1: torch_ingress_num_1      output nn_custom      input1
#> 2: torch_ingress_num_2      output nn_custom      input2

# The new task has both Sepal and Petal features
identical(mds_out[[1L]]$task, mds_out[[2L]]$task)
#> [1] TRUE
mds_out[[2L]]$task
#> 
#> ── <TaskClassif> (150x5): Iris Flowers ─────────────────────────────────────────
#> • Target: Species
#> • Target classes: setosa (33%), versicolor (33%), virginica (33%)
#> • Properties: multiclass
#> • Features (4):
#>   • dbl (4): Petal.Length, Petal.Width, Sepal.Length, Sepal.Width

# The new ingress slot contains all ingressors
identical(mds_out[[1L]]$ingress, mds_out[[2L]]$ingress)
#> [1] TRUE
mds_out[[1L]]$ingress
#> $torch_ingress_num_1.input
#> Ingress: Task[selector_name(c("Sepal.Length", "Sepal.Width"), assert_present = TRUE)] --> Tensor(NA, 2)
#> 
#> $torch_ingress_num_2.input
#> Ingress: Task[selector_name(c("Petal.Length", "Petal.Width"), assert_present = TRUE)] --> Tensor(NA, 2)
#> 

# The pointer and pointer_shape slots are different
mds_out[[1L]]$pointer
#> [1] "nn_custom" "output1"  
mds_out[[2L]]$pointer
#> [1] "nn_custom" "output2"  

mds_out[[1L]]$pointer_shape
#> [1] NA 10
mds_out[[2L]]$pointer_shape
#> [1] NA 20

## Prediction
predict_input = list(input1 = task1, input2 = task2)
tasks_out = do.call(po_torch$predict, args = list(input = predict_input))
identical(tasks_out[[1L]], tasks_out[[2L]])
#> [1] TRUE
```
