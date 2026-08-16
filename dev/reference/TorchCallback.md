# Torch Callback

This wraps a
[`CallbackSet`](https://mlr3torch.mlr-org.com/dev/reference/mlr_callback_set.md)
and annotates it with metadata, most importantly a
[`ParamSet`](https://paradox.mlr-org.com/reference/ParamSet.html). The
callback is created for the given parameter values by calling the
`$generate()` method.

This class is usually used to configure the callback of a torch learner,
e.g. when constructing a learner of in a
[`ModelDescriptor`](https://mlr3torch.mlr-org.com/dev/reference/ModelDescriptor.md).

For a list of available callbacks, see
[`mlr3torch_callbacks`](https://mlr3torch.mlr-org.com/dev/reference/mlr3torch_callbacks.md).
To conveniently retrieve a `TorchCallback`, use
[`t_clbk()`](https://mlr3torch.mlr-org.com/dev/reference/t_clbk.md).

## Parameters

Defined by the constructor argument `param_set`. If no parameter set is
provided during construction, the parameter set is constructed by
creating a parameter for each argument of the wrapped callback, where
the parameters are then of type `ParamUty`.

## See also

Other Callback:
[`as_torch_callback()`](https://mlr3torch.mlr-org.com/dev/reference/as_torch_callback.md),
[`as_torch_callbacks()`](https://mlr3torch.mlr-org.com/dev/reference/as_torch_callbacks.md),
[`callback_set()`](https://mlr3torch.mlr-org.com/dev/reference/callback_set.md),
[`mlr3torch_callbacks`](https://mlr3torch.mlr-org.com/dev/reference/mlr3torch_callbacks.md),
[`mlr_callback_set`](https://mlr3torch.mlr-org.com/dev/reference/mlr_callback_set.md),
[`mlr_callback_set.checkpoint`](https://mlr3torch.mlr-org.com/dev/reference/mlr_callback_set.checkpoint.md),
[`mlr_callback_set.progress`](https://mlr3torch.mlr-org.com/dev/reference/mlr_callback_set.progress.md),
[`mlr_callback_set.tb`](https://mlr3torch.mlr-org.com/dev/reference/mlr_callback_set.tb.md),
[`mlr_callback_set.unfreeze`](https://mlr3torch.mlr-org.com/dev/reference/mlr_callback_set.unfreeze.md),
[`mlr_context_torch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_context_torch.md),
[`t_clbk()`](https://mlr3torch.mlr-org.com/dev/reference/t_clbk.md),
[`torch_callback()`](https://mlr3torch.mlr-org.com/dev/reference/torch_callback.md)

Other Torch Descriptor:
[`TorchDescriptor`](https://mlr3torch.mlr-org.com/dev/reference/TorchDescriptor.md),
[`TorchLoss`](https://mlr3torch.mlr-org.com/dev/reference/TorchLoss.md),
[`TorchOptimizer`](https://mlr3torch.mlr-org.com/dev/reference/TorchOptimizer.md),
[`as_torch_callbacks()`](https://mlr3torch.mlr-org.com/dev/reference/as_torch_callbacks.md),
[`as_torch_loss()`](https://mlr3torch.mlr-org.com/dev/reference/as_torch_loss.md),
[`as_torch_optimizer()`](https://mlr3torch.mlr-org.com/dev/reference/as_torch_optimizer.md),
[`mlr3torch_losses`](https://mlr3torch.mlr-org.com/dev/reference/mlr3torch_losses.md),
[`mlr3torch_optimizers`](https://mlr3torch.mlr-org.com/dev/reference/mlr3torch_optimizers.md),
[`t_clbk()`](https://mlr3torch.mlr-org.com/dev/reference/t_clbk.md),
[`t_loss()`](https://mlr3torch.mlr-org.com/dev/reference/t_loss.md),
[`t_opt()`](https://mlr3torch.mlr-org.com/dev/reference/t_opt.md)

## Super class

[`TorchDescriptor`](https://mlr3torch.mlr-org.com/dev/reference/TorchDescriptor.md)
-\> `TorchCallback`

## Active bindings

- `weight`:

  (`numeric(1)` or `NULL`)  
  Overwrites the `$weight` of the generated
  [`CallbackSet`](https://mlr3torch.mlr-org.com/dev/reference/mlr_callback_set.md),
  see its section *Ordering*.

## Methods

### Public methods

- [`TorchCallback$new()`](#method-TorchCallback-initialize)

- [`TorchCallback$generate()`](#method-TorchCallback-generate)

- [`TorchCallback$clone()`](#method-TorchCallback-clone)

Inherited methods

- [`TorchDescriptor$help()`](https://mlr3torch.mlr-org.com/dev/reference/TorchDescriptor.html#method-help)
- [`TorchDescriptor$print()`](https://mlr3torch.mlr-org.com/dev/reference/TorchDescriptor.html#method-print)

------------------------------------------------------------------------

### `TorchCallback$new()`

Creates a new instance of this
[R6](https://r6.r-lib.org/reference/R6Class.html) class.

#### Usage

    TorchCallback$new(
      callback_generator,
      param_set = NULL,
      id = NULL,
      label = NULL,
      packages = NULL,
      man = NULL,
      additional_args = NULL,
      weight = NULL
    )

#### Arguments

- `callback_generator`:

  (`R6ClassGenerator`)  
  The class generator for the callback that is being wrapped.

- `param_set`:

  (`ParamSet` or `NULL`)  
  The parameter set. If `NULL` (default) it is inferred from
  `callback_generator`.

- `id`:

  (`character(1)`)  
  The id for of the new object.

- `label`:

  (`character(1)`)  
  Label for the new instance.

- `packages`:

  ([`character()`](https://rdrr.io/r/base/character.html))  
  The R packages this object depends on.

- `man`:

  (`character(1)`)  
  String in the format `[pkg]::[topic]` pointing to a manual page for
  this object. The referenced help package can be opened via method
  `$help()`.

- `additional_args`:

  (`any`)  
  Additional arguments if necessary. For learning rate schedulers, this
  is the torch::LRScheduler.

- `weight`:

  (`numeric(1)` or `NULL`)  
  Overwrites the `$weight` of the generated
  [`CallbackSet`](https://mlr3torch.mlr-org.com/dev/reference/mlr_callback_set.md),
  see its section *Ordering*. If `NULL` (default), the callback's own
  weight is kept.

------------------------------------------------------------------------

### `TorchCallback$generate()`

Generates the
[`CallbackSet`](https://mlr3torch.mlr-org.com/dev/reference/mlr_callback_set.md),
applying `$weight` if it is set.

#### Usage

    TorchCallback$generate()

------------------------------------------------------------------------

### `TorchCallback$clone()`

The objects of this class are cloneable with this method.

#### Usage

    TorchCallback$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.

## Examples

``` r
# Create a new torch callback from an existing callback set
torch_callback = TorchCallback$new(CallbackSetCheckpoint)
# The parameters are inferred
torch_callback$param_set
#> <ParamSet(2)>
#>        id    class lower upper nlevels        default  value
#>    <char>   <char> <num> <num>   <num>         <list> <list>
#> 1:   path ParamUty    NA    NA     Inf <NoDefault[0]> [NULL]
#> 2:   freq ParamUty    NA    NA     Inf <NoDefault[0]> [NULL]

# Retrieve a torch callback from the dictionary
torch_callback = t_clbk("checkpoint",
  path = tempfile(), freq = 1
)
torch_callback
#> <TorchCallback:checkpoint> Checkpoint
#> * Generator: CallbackSetCheckpoint
#> * Parameters: path=/tmp/RtmpTwMqNF/file1d6d2ff099e7, freq=1
#> * Packages: mlr3torch,torch
torch_callback$label
#> [1] "Checkpoint"
torch_callback$id
#> [1] "checkpoint"

# open the help page of the wrapped callback set
# torch_callback$help()

# Create the callback set
callback = torch_callback$generate()
callback
#> <CallbackSetCheckpoint>
#> * Stages: on_epoch_end, on_exit
# is the same as
CallbackSetCheckpoint$new(
  path = tempfile(), freq = 1
)
#> <CallbackSetCheckpoint>
#> * Stages: on_epoch_end, on_exit

# Use in a learner
learner = lrn("regr.mlp", callbacks = t_clbk("checkpoint"))
# the parameters of the callback are added to the learner's parameter set
learner$param_set
#> <ParamSetCollection(41)>
#>                       id    class lower upper nlevels        default
#>                   <char>   <char> <num> <num>   <num>         <list>
#>  1:               epochs ParamInt 0e+00   Inf     Inf <NoDefault[0]>
#>  2:               device ParamFct    NA    NA      12 <NoDefault[0]>
#>  3:          num_threads ParamInt 1e+00   Inf     Inf <NoDefault[0]>
#>  4:  num_interop_threads ParamInt 1e+00   Inf     Inf <NoDefault[0]>
#>  5:                 seed ParamInt  -Inf   Inf     Inf <NoDefault[0]>
#>  6:            eval_freq ParamInt 1e+00   Inf     Inf <NoDefault[0]>
#>  7:       measures_train ParamUty    NA    NA     Inf <NoDefault[0]>
#>  8:       measures_valid ParamUty    NA    NA     Inf <NoDefault[0]>
#>  9:             patience ParamInt 0e+00   Inf     Inf <NoDefault[0]>
#> 10:            min_delta ParamDbl 0e+00   Inf     Inf <NoDefault[0]>
#> 11: restore_best_weights ParamLgl    NA    NA       2 <NoDefault[0]>
#> 12:           batch_size ParamInt 1e+00   Inf     Inf <NoDefault[0]>
#> 13:   batch_size_predict ParamInt 1e+00   Inf     Inf <NoDefault[0]>
#> 14:              shuffle ParamLgl    NA    NA       2          FALSE
#> 15:              sampler ParamUty    NA    NA     Inf <NoDefault[0]>
#> 16:        batch_sampler ParamUty    NA    NA     Inf <NoDefault[0]>
#> 17:          num_workers ParamInt 0e+00   Inf     Inf              0
#> 18:           collate_fn ParamUty    NA    NA     Inf         [NULL]
#> 19:           pin_memory ParamLgl    NA    NA       2          FALSE
#> 20:            drop_last ParamLgl    NA    NA       2          FALSE
#> 21:              timeout ParamDbl  -Inf   Inf     Inf             -1
#> 22:       worker_init_fn ParamUty    NA    NA     Inf <NoDefault[0]>
#> 23:       worker_globals ParamUty    NA    NA     Inf <NoDefault[0]>
#> 24:      worker_packages ParamUty    NA    NA     Inf <NoDefault[0]>
#> 25:       tensor_dataset ParamFct    NA    NA       1 <NoDefault[0]>
#> 26:            jit_trace ParamLgl    NA    NA       2 <NoDefault[0]>
#> 27:              neurons ParamUty    NA    NA     Inf <NoDefault[0]>
#> 28:                    p ParamDbl 0e+00     1     Inf <NoDefault[0]>
#> 29:             n_layers ParamInt 1e+00   Inf     Inf <NoDefault[0]>
#> 30:           activation ParamUty    NA    NA     Inf <NoDefault[0]>
#> 31:      activation_args ParamUty    NA    NA     Inf <NoDefault[0]>
#> 32:                shape ParamUty    NA    NA     Inf <NoDefault[0]>
#> 33:               opt.lr ParamDbl 0e+00   Inf     Inf          0.001
#> 34:            opt.betas ParamUty    NA    NA     Inf    0.900,0.999
#> 35:              opt.eps ParamDbl 1e-16   Inf     Inf          1e-08
#> 36:     opt.weight_decay ParamDbl 0e+00   Inf     Inf              0
#> 37:          opt.amsgrad ParamLgl    NA    NA       2          FALSE
#> 38:     opt.param_groups ParamUty    NA    NA     Inf <NoDefault[0]>
#> 39:       loss.reduction ParamFct    NA    NA       2           mean
#> 40:   cb.checkpoint.path ParamUty    NA    NA     Inf <NoDefault[0]>
#> 41:   cb.checkpoint.freq ParamInt 1e+00   Inf     Inf <NoDefault[0]>
#>                       id    class lower upper nlevels        default
#>                   <char>   <char> <num> <num>   <num>         <list>
#>            value
#>           <list>
#>  1:       [NULL]
#>  2:         auto
#>  3:            1
#>  4:       [NULL]
#>  5:       random
#>  6:            1
#>  7:    <list[0]>
#>  8:    <list[0]>
#>  9:            0
#> 10:            0
#> 11:        FALSE
#> 12:       [NULL]
#> 13:       [NULL]
#> 14:         TRUE
#> 15:       [NULL]
#> 16:       [NULL]
#> 17:       [NULL]
#> 18:       [NULL]
#> 19:       [NULL]
#> 20:       [NULL]
#> 21:       [NULL]
#> 22:       [NULL]
#> 23:       [NULL]
#> 24:       [NULL]
#> 25:        FALSE
#> 26:        FALSE
#> 27:             
#> 28:          0.1
#> 29:       [NULL]
#> 30: <nn_relu[1]>
#> 31:    <list[0]>
#> 32:       [NULL]
#> 33:       [NULL]
#> 34:       [NULL]
#> 35:       [NULL]
#> 36:       [NULL]
#> 37:       [NULL]
#> 38:       [NULL]
#> 39:       [NULL]
#> 40:       [NULL]
#> 41:       [NULL]
#>            value
#>           <list>
```
