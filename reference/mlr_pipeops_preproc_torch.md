# Base Class for Lazy Tensor Preprocessing

This `PipeOp` can be used to preprocess (one or more)
[`lazy_tensor`](https://mlr3torch.mlr-org.com/reference/lazy_tensor.md)
columns contained in an
[`mlr3::Task`](https://mlr3.mlr-org.com/reference/Task.html). The
preprocessing function is specified as construction argument `fn` and
additional arguments to this function can be defined through the
`PipeOp`'s parameter set. The preprocessing is done per column, i.e. the
number of lazy tensor output columns is equal to the number of lazy
tensor input columns.

To create custom preprocessing `PipeOp`s you can use
[`pipeop_preproc_torch`](https://mlr3torch.mlr-org.com/reference/pipeop_preproc_torch.md).

## Inheriting

In addition to specifying the construction arguments, you can overwrite
the private `.shapes_out()` method. If you don't overwrite it, the
output shapes are assumed to be unknown (`NULL`).

- `.shapes_out(shapes_in, param_vals, task)`  
  ([`list()`](https://rdrr.io/r/base/list.html),
  `list(), `Task`or`NULL`) -> `list()`\cr This private method calculates the output shapes of the lazy tensor columns that are created from applying the preprocessing function with the provided parameter values (`param_vals`). The `task`is very rarely needed, but if it is it should be checked that it is not`NULL\`.

  This private method only has the responsibility to calculate the
  output shapes for one input column, i.e. the input `shapes_in` can be
  assumed to have exactly one shape vector for which it must calculate
  the output shapes and return it as a
  [`list()`](https://rdrr.io/r/base/list.html) of length 1. It can also
  be assumed that the shape is not `NULL` (i.e. unknown). Also, the
  first dimension can be `NA`, i.e. is unknown (as for the batch
  dimension).

## Input and Output Channels

See
[`PipeOpTaskPreproc`](https://mlr3pipelines.mlr-org.com/reference/PipeOpTaskPreproc.html).

## State

In addition to state elements from
[`PipeOpTaskPreprocSimple`](https://mlr3pipelines.mlr-org.com/reference/PipeOpTaskPreprocSimple.html),
the state also contains the `$param_vals` that were set during training.

## Parameters

In addition to the parameters inherited from
[`PipeOpTaskPreproc`](https://mlr3pipelines.mlr-org.com/reference/PipeOpTaskPreproc.html)
as well as those specified during construction as the argument
`param_set` there are the following parameters:

- `stages` :: `character(1)`  
  The stages during which to apply the preprocessing. Can be one of
  `"train"`, `"predict"` or `"both"`. The initial value of this
  parameter is set to `"train"` when the `PipeOp`'s id starts with
  `"augment_"` and to `"both"` otherwise. Note that the preprocessing
  that is applied during `$predict()` uses the parameters that were set
  during `$train()` and not those that are set when performing the
  prediction.

## Internals

During `$train()` / `$predict()`, a
[`PipeOpModule`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_module.md)
with one input and one output channel is created. The pipeop applies the
function `fn` to the input tensor while additionally passing the
parameter values (minus `stages` and `affect_columns`) to `fn`. The
preprocessing graph of the lazy tensor columns is shallowly cloned and
the `PipeOpModule` is added. This is done to avoid modifying user input
and means that identical `PipeOpModule`s can be part of different
preprocessing graphs. This is only possible, because the created
`PipeOpModule` is stateless.

At a later point in the graph, preprocessing graphs will be merged if
possible to avoid unnecessary computation. This is best illustrated by
example: One lazy tensor column's preprocessing graph is `A -> B`. Then,
two branches are created `B -> C` and `B -> D`, creating two
preprocessing graphs `A -> B -> C` and `A -> B -> D`. When loading the
data, we want to run the preprocessing only once, i.e. we don't want to
run the `A -> B` part twice. For this reason,
[`task_dataset()`](https://mlr3torch.mlr-org.com/reference/task_dataset.md)
will try to merge graphs and cache results from graphs. However, only
graphs using the same dataset can currently be merged.

Also, the shapes created during `$train()` and `$predict()` might
differ. To avoid the creation of graphs where the predict shapes are
incompatible with the train shapes, the hypothetical predict shapes are
already calculated during `$train()` (this is why the parameters that
are set during train are also used during predict) and the
[`PipeOpTorchModel`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_torch_model.md)
will check the train and predict shapes for compatibility before
starting the training.

Otherwise, this mechanism is very similar to the
[`ModelDescriptor`](https://mlr3torch.mlr-org.com/reference/ModelDescriptor.md)
construct.

## Super classes

[`mlr3pipelines::PipeOp`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html)
-\>
[`mlr3pipelines::PipeOpTaskPreproc`](https://mlr3pipelines.mlr-org.com/reference/PipeOpTaskPreproc.html)
-\> `PipeOpTaskPreprocTorch`

## Active bindings

- `fn`:

  The preprocessing function.

- `rowwise`:

  Whether the preprocessing is applied rowwise.

## Methods

### Public methods

- [`PipeOpTaskPreprocTorch$new()`](#method-PipeOpTaskPreprocTorch-new)

- [`PipeOpTaskPreprocTorch$shapes_out()`](#method-PipeOpTaskPreprocTorch-shapes_out)

- [`PipeOpTaskPreprocTorch$clone()`](#method-PipeOpTaskPreprocTorch-clone)

Inherited methods

- [`mlr3pipelines::PipeOp$help()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-help)
- [`mlr3pipelines::PipeOp$predict()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-predict)
- [`mlr3pipelines::PipeOp$print()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-print)
- [`mlr3pipelines::PipeOp$train()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-train)

------------------------------------------------------------------------

### Method `new()`

Creates a new instance of this
[`R6`](https://r6.r-lib.org/reference/R6Class.html) class.

#### Usage

    PipeOpTaskPreprocTorch$new(
      fn,
      id = "preproc_torch",
      param_vals = list(),
      param_set = ps(),
      packages = character(0),
      rowwise = FALSE,
      stages_init = NULL,
      tags = NULL
    )

#### Arguments

- `fn`:

  (`function` or `character(2)`)  
  The preprocessing function. Must not modify its input in-place. If it
  is a `character(2)`, the first element should be the namespace and the
  second element the name. When the preprocessing function is applied to
  the tensor, the tensor will be passed by position as the first
  argument. If the `param_set` is inferred (left as `NULL`) it is
  assumed that the first argument is the `torch_tensor`.

- `id`:

  (`character(1)`)  
  The id for of the new object.

- `param_vals`:

  (named [`list()`](https://rdrr.io/r/base/list.html))  
  Parameter values to be set after construction.

- `param_set`:

  ([`ParamSet`](https://paradox.mlr-org.com/reference/ParamSet.html))  
  In case the function `fn` takes additional parameter besides a
  [`torch_tensor`](https://torch.mlverse.org/docs/reference/torch_tensor.html)
  they can be specfied as parameters. None of the parameters can have
  the `"predict"` tag. All tags should include `"train"`.

- `packages`:

  ([`character()`](https://rdrr.io/r/base/character.html))  
  The packages the preprocessing function depends on.

- `rowwise`:

  (`logical(1)`)  
  Whether the preprocessing function is applied rowwise (and then
  concatenated by row) or directly to the whole tensor. In the first
  case there is no batch dimension.

- `stages_init`:

  (`character(1)`)  
  Initial value for the `stages` parameter.

- `tags`:

  ([`character()`](https://rdrr.io/r/base/character.html))  
  Tags for the pipeop.

------------------------------------------------------------------------

### Method `shapes_out()`

Calculates the output shapes that would result in applying the
preprocessing to one or more lazy tensor columns with the provided
shape. Names are ignored and only order matters. It uses the parameter
values that are currently set.

#### Usage

    PipeOpTaskPreprocTorch$shapes_out(shapes_in, stage = NULL, task = NULL)

#### Arguments

- `shapes_in`:

  ([`list()`](https://rdrr.io/r/base/list.html) of
  ([`integer()`](https://rdrr.io/r/base/integer.html) or `NULL`))  
  The input input shapes of the lazy tensors. `NULL` indicates that the
  shape is unknown. First dimension must be `NA` (if it is not `NULL`).

- `stage`:

  (`character(1)`)  
  The stage: either `"train"` or `"predict"`.

- `task`:

  ([`Task`](https://mlr3.mlr-org.com/reference/Task.html) or `NULL`)  
  The task, which is very rarely needed.

#### Returns

[`list()`](https://rdrr.io/r/base/list.html) of
([`integer()`](https://rdrr.io/r/base/integer.html) or `NULL`)

------------------------------------------------------------------------

### Method `clone()`

The objects of this class are cloneable with this method.

#### Usage

    PipeOpTaskPreprocTorch$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.

## Examples

``` r
# Creating a simple task
d = data.table(
  x1 = as_lazy_tensor(matrix(rnorm(10), ncol = 1)),
  x2 = as_lazy_tensor(matrix(rnorm(10), ncol = 1)),
  x3 = as_lazy_tensor(matrix(as.double(1:10), ncol = 1)),
  y = rnorm(10)
)

taskin = as_task_regr(d, target = "y")

# Creating a simple preprocessing pipeop
po_simple = po("preproc_torch",
  # get rid of environment baggage
  fn = mlr3misc::crate(function(x, a) x + a),
  param_set = paradox::ps(a = paradox::p_int(tags = c("train", "required")))
)

po_simple$param_set$set_values(
  a = 100,
  affect_columns = selector_name(c("x1", "x2")),
  stages = "both" # use during train and predict
)

taskout_train = po_simple$train(list(taskin))[[1L]]
materialize(taskout_train$data(cols = c("x1", "x2")), rbind = TRUE)
#> $x1
#> torch_tensor
#>   98.8255
#>   98.2403
#>  100.0584
#>  101.1645
#>  100.3376
#>   98.9455
#>  100.6608
#>   99.1766
#>  100.4370
#>  100.3724
#> [ CPUFloatType{10,1} ]
#> 
#> $x2
#> torch_tensor
#>   98.3532
#>   98.0764
#>  100.3808
#>  101.3757
#>  100.8259
#>   99.5839
#>  100.9821
#>  100.2422
#>   99.0621
#>  101.5059
#> [ CPUFloatType{10,1} ]
#> 

taskout_predict_noaug = po_simple$predict(list(taskin))[[1L]]
materialize(taskout_predict_noaug$data(cols = c("x1", "x2")), rbind = TRUE)
#> $x1
#> torch_tensor
#>   98.8255
#>   98.2403
#>  100.0584
#>  101.1645
#>  100.3376
#>   98.9455
#>  100.6608
#>   99.1766
#>  100.4370
#>  100.3724
#> [ CPUFloatType{10,1} ]
#> 
#> $x2
#> torch_tensor
#>   98.3532
#>   98.0764
#>  100.3808
#>  101.3757
#>  100.8259
#>   99.5839
#>  100.9821
#>  100.2422
#>   99.0621
#>  101.5059
#> [ CPUFloatType{10,1} ]
#> 

po_simple$param_set$set_values(
  stages = "train"
)

# transformation is not applied
taskout_predict_aug = po_simple$predict(list(taskin))[[1L]]
materialize(taskout_predict_aug$data(cols = c("x1", "x2")), rbind = TRUE)
#> $x1
#> torch_tensor
#>   98.8255
#>   98.2403
#>  100.0584
#>  101.1645
#>  100.3376
#>   98.9455
#>  100.6608
#>   99.1766
#>  100.4370
#>  100.3724
#> [ CPUFloatType{10,1} ]
#> 
#> $x2
#> torch_tensor
#>   98.3532
#>   98.0764
#>  100.3808
#>  101.3757
#>  100.8259
#>   99.5839
#>  100.9821
#>  100.2422
#>   99.0621
#>  101.5059
#> [ CPUFloatType{10,1} ]
#> 

# Creating a more complex preprocessing PipeOp
PipeOpPreprocTorchPoly = R6::R6Class("PipeOpPreprocTorchPoly",
 inherit = PipeOpTaskPreprocTorch,
 public = list(
   initialize = function(id = "preproc_poly", param_vals = list()) {
     param_set = paradox::ps(
       n_degree = paradox::p_int(lower = 1L, tags = c("train", "required"))
     )
     param_set$set_values(
       n_degree = 1L
     )
     fn = mlr3misc::crate(function(x, n_degree) {
       torch::torch_cat(
         lapply(seq_len(n_degree), function(d) torch::torch_pow(x, d)),
         dim = 2L
       )
     })

     super$initialize(
       fn = fn,
       id = id,
       packages = character(0),
       param_vals = param_vals,
       param_set = param_set,
       stages_init = "both"
     )
   }
 ),
 private = list(
   .shapes_out = function(shapes_in, param_vals, task) {
     # shapes_in is a list of length 1 containing the shapes
     checkmate::assert_true(length(shapes_in[[1L]]) == 2L)
     if (shapes_in[[1L]][2L] != 1L) {
       stop("Input shape must be (NA, 1)")
     }
     list(c(NA, param_vals$n_degree))
   }
 )
)

po_poly = PipeOpPreprocTorchPoly$new(
  param_vals = list(n_degree = 3L, affect_columns = selector_name("x3"))
)

po_poly$shapes_out(list(c(NA, 1L)), stage = "train")
#> [[1]]
#> [1] NA  3
#> 

taskout = po_poly$train(list(taskin))[[1L]]
materialize(taskout$data(cols = "x3"), rbind = TRUE)
#> $x3
#> torch_tensor
#>     1     1     1
#>     2     4     8
#>     3     9    27
#>     4    16    64
#>     5    25   125
#>     6    36   216
#>     7    49   343
#>     8    64   512
#>     9    81   729
#>    10   100  1000
#> [ CPUFloatType{10,3} ]
#> 
```
