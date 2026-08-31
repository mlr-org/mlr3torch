# Create a PipeOpTorch

Helper function to create a custom
[`PipeOpTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch.md)
class for the most common cases. A practical guide to this function is
the article [Writing your own
PipeOpTorch](https://mlr3torch.mlr-org.com/articles/custom_pipeop_torch.html).
For more information and the more general case, see the *Inheriting*
section of
[`PipeOpTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch.md).
The function works similarly to
[`nn_module()`](https://torch.mlverse.org/docs/reference/nn_module.html),
except that `$initialize()` can take two further arguments that the
`PipeOp` supplies: `shapes_in` and `task` and that it also needs to
implement the method `$shapes_out()` which belongs to the `PipeOpTorch`.

## Usage

``` r
pipeop_torch(
  id,
  initialize = NULL,
  forward,
  shapes_out,
  param_set = NULL,
  in_channels = NULL,
  out_channels = 1L,
  packages = character(0),
  tags = NULL,
  classname = NULL,
  parent_env = parent.frame()
)
```

## Arguments

- id:

  (`character(1)`)  
  The id for of the new object.

- initialize:

  (`function` or `NULL`)  
  The `$initialize()` method of the module. Its arguments become the
  hyperparameters of the `PipeOp`, except for `shapes_in` (the input
  shapes, named after the input channels) and `task` (the
  [`Task`](https://mlr3.mlr-org.com/reference/Task.html) or `NULL`),
  which are supplied by the `PipeOp` and passed only if the function
  declares them.

- forward:

  (`function`)  
  The `$forward()` method of the module.

- shapes_out:

  (`function`)  
  The shapes of the tensors that the module produces, as a function of
  `shapes_in`, `param_vals` and `task` – only the arguments that are
  declared are passed, so an operator whose output shape depends on
  nothing else is written as `function(shapes_in)`. Any dimension of
  `shapes_in` can be `NA`, i.e. unknown, so this function must not
  assume that a dimension it reads is known; see the *Inheriting*
  section of
  [`PipeOpTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch.md).

- param_set:

  ([`ParamSet`](https://paradox.mlr-org.com/reference/ParamSet.html) or
  `NULL`)  
  The parameter set. If left as `NULL` (default), it is inferred from
  the arguments of `initialize`: All arguments but `shapes_in` and
  `task` become an untyped parameter tagged `"train"`, and additionally
  `"required"` if it has no default.

- in_channels:

  ([`character()`](https://rdrr.io/r/base/character.html) or
  `integer(1)` or `NULL`)  
  The input channels, either as names or as a count, where `0` means a
  single *vararg* channel. If `NULL` (default), the arguments of
  `forward` are used.

- out_channels:

  ([`character()`](https://rdrr.io/r/base/character.html) or
  `integer(1)`)  
  The output channels, either as names or as a count, `1` by default. A
  module with more than one output channel must return a
  [`list()`](https://rdrr.io/r/base/list.html), in the order of the
  channels.

- packages:

  ([`character()`](https://rdrr.io/r/base/character.html))  
  The R packages this object depends on.

- tags:

  ([`character()`](https://rdrr.io/r/base/character.html))  
  Tags for the `PipeOp`. The tag `"torch"` is always added.

- classname:

  (`character(1)`)  
  The class name of the generated
  [`R6Class`](https://r6.r-lib.org/reference/R6Class.html). By default
  it is derived from `id`: a leading `"nn_"` is dropped, the remaining
  `_`-separated words are capitalized and pasted together, and
  `"PipeOpTorch"` is prepended, so the id `"nn_scale"` gives
  `"PipeOpTorchScale"`.

- parent_env:

  (`environment`)  
  The environment in which the module's methods are evaluated, the
  calling environment by default, as for
  [`nn_module()`](https://torch.mlverse.org/docs/reference/nn_module.html).
  `initialize` and `forward` become methods of the module and are
  therefore evaluated in this environment rather than in the one they
  were written in; `shapes_out`, which is not a method of the module,
  keeps its own environment. The two only differ when the functions are
  written somewhere other than the caller of `pipeop_torch()`, e.g. in a
  function that wraps it.

## Value

An [`R6Class`](https://r6.r-lib.org/reference/R6Class.html) generator
inheriting from
[`PipeOpTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch.md).

## See also

Other Graph Network:
[`ModelDescriptor()`](https://mlr3torch.mlr-org.com/dev/reference/ModelDescriptor.md),
[`TorchIngressToken()`](https://mlr3torch.mlr-org.com/dev/reference/TorchIngressToken.md),
[`as_learner_torch()`](https://mlr3torch.mlr-org.com/dev/reference/as_learner_torch.md),
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
# A layer that scales its input by a learned factor
# Note that the number of features is read from the input shape
PipeOpTorchCustomScale = pipeop_torch("nn_custom_scale",
  initialize = function(shapes_in, init = 1) {
    self$weight = nn_parameter(torch_full(tail(shapes_in[[1L]], 1L), init))
  },
  forward = function(input) input * self$weight,
  shapes_out = function(shapes_in) shapes_in # scaling leaves the shape as it is
)

po_custom_scale = PipeOpTorchCustomScale$new()
# `init` is a hyperparameter, the number of features is not
po_custom_scale$param_set$ids()
#> [1] "init"
po_custom_scale$shapes_out(list(c(NA, 4)))
#> $output
#> [1] NA  4
#> 
# the operator can now be used like any other, and the module is built with 4 features
md = po("torch_ingress_num") %>>% po_custom_scale %>>% po("nn_head")
network = model_descriptor_to_module(md$train(tsk("iris"))[[1L]])
network
#> An `nn_module` containing 19 parameters.
#> 
#> ── Modules ─────────────────────────────────────────────────────────────────────
#> • module_list: <nn_module_list> #19 parameters

# To use it via `nn("custom_scale")` or `po("nn_custom_scale")` we could run the line below:
# mlr_pipeops$add("nn_custom_scale", PipeOpTorchCustomScale)
```
