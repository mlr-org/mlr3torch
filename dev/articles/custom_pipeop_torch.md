# Writing your own PipeOpTorch

The {mlr3torch} package already provides `PipeOpTorch` objects for many
common neural network layers. However, sooner or later you will want to
add your own custom layer. This article will show you how to do so. We
assume that you already understand how to use existing `PipeOpTorch`
objects, which is covered in [Defining an
Architecture](https://mlr3torch.mlr-org.com/dev/articles/pipeop_torch.md).

A `PipeOpTorch` is a descriptor of an `nn_module_generator`: it does not
contain a module, it says how to build one once the shape of its input
is known. We will first create a custom module generator and then turn
it into a `PipeOpTorch`. As an example we re-implement
[`torch::nn_linear()`](https://torch.mlverse.org/docs/reference/nn_linear.html).

## A module generator

A module generator is created with
[`torch::nn_module()`](https://torch.mlverse.org/docs/reference/nn_module.html)
and it requires implementing two methods:

1.  `$initialize()`, which constructs the instance, and
2.  `$forward()`, which runs the constructed layer on some tensor
    inputs.

``` r

nn_my_linear = nn_module("nn_my_linear",
  initialize = function(in_features, out_features) {
    self$weight = nn_parameter(torch_randn(out_features, in_features) / sqrt(in_features))
    self$bias = nn_parameter(torch_zeros(out_features))
  },
  forward = function(input) {
    nnf_linear(input, self$weight, self$bias)
  }
)
```

Calling the generator builds a module instance (via `$initialize()` )
and calling the resulting module on an input tensor runs its
`$forward()` method:

``` r

layer = nn_my_linear(in_features = 4, out_features = 2)
layer(torch_randn(3, 4))
#> torch_tensor
#>  0.0748  0.1402
#>  0.5230 -0.2631
#>  0.3619 -0.0215
#> [ CPUFloatType{3,2} ][ grad_fn = <AddmmBackward0> ]
```

## The same layer as a `PipeOpTorch`

For `nn_my_linear`, the `in_features` need to be manually specified.
However, they are an *auxiliary* parameter, in the sense that they can
be inferred from the input shape that the layer will receive during
runtime. The core idea of a `PipeOpTorch` is that it does not require
specifying such auxiliary parameters but that they are automatically
determined. To enable this, a `PipeOpTorch` needs to define how to infer
these auxiliary parameters and how to compute the output shapes for the
specified input shapes and hyperparameters. In the code below, the
`$initialize()` method of the module generator takes in the `shapes_in`
and computes the `in_features` from their last dimension. The shape
inference is implemented via the `$shapes_out()` method. Note that the
input shapes can have unknown dimensions, which are encoded as being
`NA`, so we need to assert that the feature dimension which we use to
set `in_features` is actually known and also that the number of input
dimensions is at least 2. The {mlr3torch} package also comes with other
such assertion helpers that are useful when implementing the shape
inference. Note that when pipeop instances are `$train()`ed,
`$shapes_out()` runs before `$initialize()`, so it is enough to have the
input shape assertions only in `$shapes_out()`.

``` r

PipeOpTorchMyLinear = pipeop_torch("nn_my_linear",
  initialize = function(shapes_in, out_features) {
    in_features = tail(shapes_in[[1L]], 1L)
    self$weight = nn_parameter(torch_randn(out_features, in_features) / sqrt(in_features))
    self$bias = nn_parameter(torch_zeros(out_features))
  },
  forward = function(input) {
    nnf_linear(input, self$weight, self$bias)
  },
  shapes_out = function(shapes_in, param_vals) {
    shape_in = shapes_in[[1L]]
    assert_ndim(shape_in, min = 2L)
    assert_known_dims(shape_in, length(shape_in))
    list(c(head(shape_in, -1L), param_vals$out_features))
  }
)
```

The only non-auxiliary hyperparameter of our module is `out_features`,
which can be set via the pipeop’s parameter set, which by default is
inferred from the module’s `$initialize()` method.

``` r

po_linear = PipeOpTorchMyLinear$new(param_vals = list(out_features = 2))
po_linear$param_set$ids()
#> [1] "out_features"
po_linear$shapes_out(list(c(NA, 4)))
#> $output
#> [1] NA  2

md = (po("torch_ingress_num") %>>% po_linear)$train(tsk("iris"))[[1L]]
model_descriptor_to_module(md)
#> An `nn_module` containing 10 parameters.
#> 
#> ── Modules ─────────────────────────────────────────────────────────────────────
#> • module_list: <nn_module_list> #10 parameters
```

In order to use the pipeop via the common `nn("my_linear")`, it needs to
be registered:

``` r

mlr_pipeops$add("nn_my_linear", PipeOpTorchMyLinear)
nn("my_linear")
#> 
#> ── PipeOp <my_linear>: not trained ─────────────────────────────────────────────
#> Values: list()
#> 
#> ── Input channels: 
#>    name           train predict
#>  <char>          <char>  <char>
#>   input ModelDescriptor    Task
#> 
#> ── Output channels: 
#>    name           train predict
#>  <char>          <char>  <char>
#>  output ModelDescriptor    Task
```

By default, `pipeop_torch` assumes that the module returns a single
tensor. If this is not the case, you can specify the `out_channels` to
either be the number of output channels of the pipeop or by providing
explicit channel names. Also, you can specify the parameter set of the
`PipeOp` explicitly, which allows to annotate their types and admissible
ranges.

While `pipeop_torch` covers most of the cases, explicitly implementing a
`PipeOpTorch` as an `R6Class` comes with more flexibility, which can
come in handy in more complex cases and which we will cover next.

## Inheriting explicitly from `PipeOpTorch`

Under the hood, `pipeop_torch` creates an `R6Class` that inherits from
`PipeOpTorch`. Next, we will show how to do this manually, starting from
the `nn_my_linear` from above. For more information and details, also
see the *Inheriting* section of `PipeOpTorch`.

In the code below, we specify the parameter set, how to infer the
`in_features` from the input shapes (via
`private$.shape_dependent_params()`), as well as the shape inference
(via `private$.shapes_out()`). The signature of the two private methods
is:

- `.shape_dependent_params(shapes_in, param_vals, task)` returns *all*
  arguments with which the module generator is called, i.e. the
  user-specified `param_vals` plus the auxiliary parameters that are
  inferred from the input shapes.
- `.shapes_out(shapes_in, param_vals, task)` returns the output shapes,
  just like the `shapes_out` argument of
  [`pipeop_torch()`](https://mlr3torch.mlr-org.com/dev/reference/pipeop_torch.md),
  except that it always receives all three arguments and not only those
  that it declares.

``` r

library(paradox)

PipeOpTorchMyLinear = R6::R6Class("PipeOpTorchMyLinear",
  inherit = PipeOpTorch,
  public = list(
    initialize = function(id = "nn_my_linear", param_vals = list()) {
      param_set = ps(
        out_features = p_int(lower = 1L, tags = c("required", "train"))
      )
      super$initialize(
        id = id,
        param_vals = param_vals,
        param_set = param_set,
        module_generator = nn_my_linear
      )
    }
  ),
  private = list(
    .shape_dependent_params = function(shapes_in, param_vals, task) {
      c(param_vals, list(in_features = tail(shapes_in[[1L]], 1L)))
    },
    .shapes_out = function(shapes_in, param_vals, task) {
      shape_in = shapes_in[[1L]]
      assert_ndim(shape_in, min = 2L)
      assert_known_dims(shape_in, length(shape_in))
      list(c(head(shape_in, -1L), param_vals$out_features))
    }
  )
)
```

Because `$shapes_out()` is called before the module is constructed, it
is again enough to assert the input shapes in `.shapes_out()` only. The
resulting `PipeOp` behaves just like the generated one:

``` r

po_linear = PipeOpTorchMyLinear$new(param_vals = list(out_features = 2))
po_linear$shapes_out(list(c(NA, 4)))
#> $output
#> [1] NA  2

md = (po("torch_ingress_num") %>>% po_linear)$train(tsk("iris"))[[1L]]
model_descriptor_to_module(md)
#> An `nn_module` containing 10 parameters.
#> 
#> ── Modules ─────────────────────────────────────────────────────────────────────
#> • module_list: <nn_module_list> #10 parameters
```

Registering the class in the `mlr_pipeops` dictionary works exactly as
above, i.e. via `mlr_pipeops$add("nn_my_linear", PipeOpTorchMyLinear)`.
If a module cannot be constructed by calling its generator with a list
of arguments, the private method
`.make_module(shapes_in, param_vals, task)`, which returns the
`nn_module`, can be overloaded instead of `.shape_dependent_params()`.

## More than one channel

A module whose `$forward()` method takes more than one tensor becomes a
`PipeOp` with more than one input channel, and one that returns more
than one tensor becomes a `PipeOp` with more than one output channel. As
an example we take a module that applies a separate linear layer to each
of its two inputs.

With
[`pipeop_torch()`](https://mlr3torch.mlr-org.com/dev/reference/pipeop_torch.md),
the input channels are inferred from the arguments of `$forward()`, so
they are called `"input1"` and `"input2"` here, and `shapes_in` is named
accordingly. Because the module returns two tensors, the output channels
have to be specified via `out_channels`, either as a count or, as below,
by their names.

``` r

PipeOpTorchParallelLinear = pipeop_torch("nn_parallel_linear",
  initialize = function(shapes_in, d_out1, d_out2, bias = TRUE) {
    self$linear1 = nn_linear(tail(shapes_in$input1, 1L), d_out1, bias)
    self$linear2 = nn_linear(tail(shapes_in$input2, 1L), d_out2, bias)
  },
  forward = function(input1, input2) {
    list(self$linear1(input1), self$linear2(input2))
  },
  out_channels = c("output1", "output2"),
  shapes_out = function(shapes_in, param_vals) {
    for (shape_in in shapes_in) {
      assert_ndim(shape_in, min = 2L)
      assert_known_dims(shape_in, length(shape_in))
    }
    list(
      c(head(shapes_in$input1, -1L), param_vals$d_out1),
      c(head(shapes_in$input2, -1L), param_vals$d_out2)
    )
  }
)

po_parallel_linear = PipeOpTorchParallelLinear$new(
  param_vals = list(d_out1 = 10, d_out2 = 20)
)
po_parallel_linear$shapes_out(list(input1 = c(NA, 2), input2 = c(NA, 2)))
#> $output1
#> [1] NA 10
#> 
#> $output2
#> [1] NA 20
```

Written as an `R6Class`, the module is again an ordinary `nn_module`
that receives its input dimensions:

``` r

nn_parallel_linear = nn_module("nn_parallel_linear",
  initialize = function(d_in1, d_in2, d_out1, d_out2, bias = TRUE) {
    self$linear1 = nn_linear(d_in1, d_out1, bias)
    self$linear2 = nn_linear(d_in2, d_out2, bias)
  },
  forward = function(input1, input2) {
    list(
      output1 = self$linear1(input1),
      output2 = self$linear2(input2)
    )
  }
)
```

The channel names are now specified via the `inname` and `outname`
arguments of `super$initialize()`. The tensors are passed to the
module’s `$forward()` method by position, so it is the *order* of the
input channels that has to match its arguments. Naming the `inname`s
after those arguments is nonetheless recommended, as it avoids any
ambiguity about which tensor ends up where. The `shapes_in` are named
after the input channels, while the shapes returned by `.shapes_out()`
are assigned to the output channels by position.

``` r

PipeOpTorchParallelLinear = R6::R6Class("PipeOpTorchParallelLinear",
  inherit = PipeOpTorch,
  public = list(
    initialize = function(id = "nn_parallel_linear", param_vals = list()) {
      param_set = ps(
        d_out1 = p_int(lower = 1L, tags = c("required", "train")),
        d_out2 = p_int(lower = 1L, tags = c("required", "train")),
        bias = p_lgl(default = TRUE, tags = "train")
      )
      super$initialize(
        id = id,
        param_vals = param_vals,
        param_set = param_set,
        inname = c("input1", "input2"),
        outname = c("output1", "output2"),
        module_generator = nn_parallel_linear
      )
    }
  ),
  private = list(
    .shape_dependent_params = function(shapes_in, param_vals, task) {
      c(param_vals, list(
        d_in1 = tail(shapes_in[["input1"]], 1L),
        d_in2 = tail(shapes_in[["input2"]], 1L)
      ))
    },
    .shapes_out = function(shapes_in, param_vals, task) {
      for (shape_in in shapes_in) {
        assert_ndim(shape_in, min = 2L)
        assert_known_dims(shape_in, length(shape_in))
      }
      list(
        c(head(shapes_in[["input1"]], -1L), param_vals$d_out1),
        c(head(shapes_in[["input2"]], -1L), param_vals$d_out2)
      )
    }
  )
)

po_parallel_linear = PipeOpTorchParallelLinear$new(
  param_vals = list(d_out1 = 10, d_out2 = 20)
)
po_parallel_linear$shapes_out(list(input1 = c(NA, 2), input2 = c(NA, 2)))
#> $output1
#> [1] NA 10
#> 
#> $output2
#> [1] NA 20
```

Note that it is also possible to dynamically determine the input and
output channels depending on the constructor arguments of the
`PipeOpTorch`. What is not possible is that the number of input or
output channels depends on specific parameter values.
