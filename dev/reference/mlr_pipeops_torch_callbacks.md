# Callback Configuration

Configures the callbacks of a deep learning model.

## Parameters

The parameters are defined dynamically from the callbacks, where the id
of the respective callbacks is the respective set id.

## Input and Output Channels

There is one input channel `"input"` and one output channel `"output"`.
During *training*, the channels are of class
[`ModelDescriptor`](https://mlr3torch.mlr-org.com/dev/reference/ModelDescriptor.md).
During *prediction*, the channels are of class
[`Task`](https://mlr3.mlr-org.com/reference/Task.html).

## State

The state is the value calculated by the public method `shapes_out()`.

## Internals

During training the callbacks are cloned and added to the
[`ModelDescriptor`](https://mlr3torch.mlr-org.com/dev/reference/ModelDescriptor.md).

## See also

Other Model Configuration:
[`ModelDescriptor()`](https://mlr3torch.mlr-org.com/dev/reference/ModelDescriptor.md),
[`mlr_pipeops_torch_loss`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch_loss.md),
[`mlr_pipeops_torch_optimizer`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch_optimizer.md),
[`model_descriptor_union()`](https://mlr3torch.mlr-org.com/dev/reference/model_descriptor_union.md)

Other PipeOp:
[`mlr_pipeops_module`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_module.md),
[`mlr_pipeops_torch_optimizer`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch_optimizer.md)

## Super class

[`mlr3pipelines::PipeOp`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html)
-\> `PipeOpTorchCallbacks`

## Methods

### Public methods

- [`PipeOpTorchCallbacks$new()`](#method-PipeOpTorchCallbacks-new)

- [`PipeOpTorchCallbacks$clone()`](#method-PipeOpTorchCallbacks-clone)

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

    PipeOpTorchCallbacks$new(
      callbacks = list(),
      id = "torch_callbacks",
      param_vals = list()
    )

#### Arguments

- `callbacks`:

  (`list` of
  [`TorchCallback`](https://mlr3torch.mlr-org.com/dev/reference/TorchCallback.md)s)  
  The callbacks (or something convertible via
  [`as_torch_callbacks()`](https://mlr3torch.mlr-org.com/dev/reference/as_torch_callbacks.md)).
  Must have unique ids. All callbacks are cloned during construction.

- `id`:

  (`character(1)`)  
  Identifier of the resulting object.

- `param_vals`:

  ([`list()`](https://rdrr.io/r/base/list.html))  
  List of hyperparameter settings, overwriting the hyperparameter
  settings that would otherwise be set during construction.

------------------------------------------------------------------------

### Method `clone()`

The objects of this class are cloneable with this method.

#### Usage

    PipeOpTorchCallbacks$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.

## Examples

``` r
po_cb = po("torch_callbacks", "checkpoint")
po_cb$param_set
#> <ParamSetCollection(3)>
#>                      id    class lower upper nlevels        default  value
#>                  <char>   <char> <num> <num>   <num>         <list> <list>
#> 1:      checkpoint.path ParamUty    NA    NA     Inf <NoDefault[0]> [NULL]
#> 2:      checkpoint.freq ParamInt     1   Inf     Inf <NoDefault[0]> [NULL]
#> 3: checkpoint.freq_type ParamFct    NA    NA       2          epoch [NULL]
mdin = po("torch_ingress_num")$train(list(tsk("iris")))
mdin[[1L]]$callbacks
#> named list()
mdout = po_cb$train(mdin)[[1L]]
mdout$callbacks
#> $checkpoint
#> <TorchCallback:checkpoint> Checkpoint
#> * Generator: CallbackSetCheckpoint
#> * Parameters: list()
#> * Packages: mlr3torch,torch
#> 
# Can be called again
po_cb1 = po("torch_callbacks", t_clbk("progress"))
mdout1 = po_cb1$train(list(mdout))[[1L]]
mdout1$callbacks
#> $progress
#> <TorchCallback:progress> Progress
#> * Generator: CallbackSetProgress
#> * Parameters: list()
#> * Packages: progress,mlr3torch,torch
#> 
#> $checkpoint
#> <TorchCallback:checkpoint> Checkpoint
#> * Generator: CallbackSetCheckpoint
#> * Parameters: list()
#> * Packages: mlr3torch,torch
#> 
```
