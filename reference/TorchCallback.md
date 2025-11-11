# Torch Callback

This wraps a
[`CallbackSet`](https://mlr3torch.mlr-org.com/reference/mlr_callback_set.md)
and annotates it with metadata, most importantly a
[`ParamSet`](https://paradox.mlr-org.com/reference/ParamSet.html). The
callback is created for the given parameter values by calling the
`$generate()` method.

This class is usually used to configure the callback of a torch learner,
e.g. when constructing a learner of in a
[`ModelDescriptor`](https://mlr3torch.mlr-org.com/reference/ModelDescriptor.md).

For a list of available callbacks, see
[`mlr3torch_callbacks`](https://mlr3torch.mlr-org.com/reference/mlr3torch_callbacks.md).
To conveniently retrieve a `TorchCallback`, use
[`t_clbk()`](https://mlr3torch.mlr-org.com/reference/t_clbk.md).

## Parameters

Defined by the constructor argument `param_set`. If no parameter set is
provided during construction, the parameter set is constructed by
creating a parameter for each argument of the wrapped loss function,
where the parametes are then of type `ParamUty`.

## See also

Other Callback:
[`as_torch_callback()`](https://mlr3torch.mlr-org.com/reference/as_torch_callback.md),
[`as_torch_callbacks()`](https://mlr3torch.mlr-org.com/reference/as_torch_callbacks.md),
[`callback_set()`](https://mlr3torch.mlr-org.com/reference/callback_set.md),
[`mlr3torch_callbacks`](https://mlr3torch.mlr-org.com/reference/mlr3torch_callbacks.md),
[`mlr_callback_set`](https://mlr3torch.mlr-org.com/reference/mlr_callback_set.md),
[`mlr_callback_set.checkpoint`](https://mlr3torch.mlr-org.com/reference/mlr_callback_set.checkpoint.md),
[`mlr_callback_set.progress`](https://mlr3torch.mlr-org.com/reference/mlr_callback_set.progress.md),
[`mlr_callback_set.tb`](https://mlr3torch.mlr-org.com/reference/mlr_callback_set.tb.md),
[`mlr_callback_set.unfreeze`](https://mlr3torch.mlr-org.com/reference/mlr_callback_set.unfreeze.md),
[`mlr_context_torch`](https://mlr3torch.mlr-org.com/reference/mlr_context_torch.md),
[`t_clbk()`](https://mlr3torch.mlr-org.com/reference/t_clbk.md),
[`torch_callback()`](https://mlr3torch.mlr-org.com/reference/torch_callback.md)

Other Torch Descriptor:
[`TorchDescriptor`](https://mlr3torch.mlr-org.com/reference/TorchDescriptor.md),
[`TorchLoss`](https://mlr3torch.mlr-org.com/reference/TorchLoss.md),
[`TorchOptimizer`](https://mlr3torch.mlr-org.com/reference/TorchOptimizer.md),
[`as_torch_callbacks()`](https://mlr3torch.mlr-org.com/reference/as_torch_callbacks.md),
[`as_torch_loss()`](https://mlr3torch.mlr-org.com/reference/as_torch_loss.md),
[`as_torch_optimizer()`](https://mlr3torch.mlr-org.com/reference/as_torch_optimizer.md),
[`mlr3torch_losses`](https://mlr3torch.mlr-org.com/reference/mlr3torch_losses.md),
[`mlr3torch_optimizers`](https://mlr3torch.mlr-org.com/reference/mlr3torch_optimizers.md),
[`t_clbk()`](https://mlr3torch.mlr-org.com/reference/t_clbk.md),
[`t_loss()`](https://mlr3torch.mlr-org.com/reference/t_loss.md),
[`t_opt()`](https://mlr3torch.mlr-org.com/reference/t_opt.md)

## Super class

[`mlr3torch::TorchDescriptor`](https://mlr3torch.mlr-org.com/reference/TorchDescriptor.md)
-\> `TorchCallback`

## Methods

### Public methods

- [`TorchCallback$new()`](#method-TorchCallback-new)

- [`TorchCallback$clone()`](#method-TorchCallback-clone)

Inherited methods

- [`mlr3torch::TorchDescriptor$generate()`](https://mlr3torch.mlr-org.com/reference/TorchDescriptor.html#method-generate)
- [`mlr3torch::TorchDescriptor$help()`](https://mlr3torch.mlr-org.com/reference/TorchDescriptor.html#method-help)
- [`mlr3torch::TorchDescriptor$print()`](https://mlr3torch.mlr-org.com/reference/TorchDescriptor.html#method-print)

------------------------------------------------------------------------

### Method `new()`

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
      additional_args = NULL
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

------------------------------------------------------------------------

### Method `clone()`

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
#> <ParamSet(3)>
#>           id    class lower upper nlevels        default  value
#>       <char>   <char> <num> <num>   <num>         <list> <list>
#> 1:      path ParamUty    NA    NA     Inf <NoDefault[0]> [NULL]
#> 2:      freq ParamUty    NA    NA     Inf <NoDefault[0]> [NULL]
#> 3: freq_type ParamUty    NA    NA     Inf <NoDefault[0]> [NULL]

# Retrieve a torch callback from the dictionary
torch_callback = t_clbk("checkpoint",
  path = tempfile(), freq = 1
)
torch_callback
#> <TorchCallback:checkpoint> Checkpoint
#> * Generator: CallbackSetCheckpoint
#> * Parameters: path=/tmp/Rtmpk6MSWZ/file7cbe761d4e41, freq=1
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
#> * Stages: on_batch_end, on_epoch_end, on_exit
# is the same as
CallbackSetCheckpoint$new(
  path = tempfile(), freq = 1
)
#> <CallbackSetCheckpoint>
#> * Stages: on_batch_end, on_epoch_end, on_exit

# Use in a learner
learner = lrn("regr.mlp", callbacks = t_clbk("checkpoint"))
# the parameters of the callback are added to the learner's parameter set
learner$param_set
#> <ParamSetCollection(40)>
#>                          id    class lower upper nlevels        default
#>                      <char>   <char> <num> <num>   <num>         <list>
#>  1:                  epochs ParamInt 0e+00   Inf     Inf <NoDefault[0]>
#>  2:                  device ParamFct    NA    NA      12 <NoDefault[0]>
#>  3:             num_threads ParamInt 1e+00   Inf     Inf <NoDefault[0]>
#>  4:     num_interop_threads ParamInt 1e+00   Inf     Inf <NoDefault[0]>
#>  5:                    seed ParamInt  -Inf   Inf     Inf <NoDefault[0]>
#>  6:               eval_freq ParamInt 1e+00   Inf     Inf <NoDefault[0]>
#>  7:          measures_train ParamUty    NA    NA     Inf <NoDefault[0]>
#>  8:          measures_valid ParamUty    NA    NA     Inf <NoDefault[0]>
#>  9:                patience ParamInt 0e+00   Inf     Inf <NoDefault[0]>
#> 10:               min_delta ParamDbl 0e+00   Inf     Inf <NoDefault[0]>
#> 11:              batch_size ParamInt 1e+00   Inf     Inf <NoDefault[0]>
#> 12:                 shuffle ParamLgl    NA    NA       2          FALSE
#> 13:                 sampler ParamUty    NA    NA     Inf <NoDefault[0]>
#> 14:           batch_sampler ParamUty    NA    NA     Inf <NoDefault[0]>
#> 15:             num_workers ParamInt 0e+00   Inf     Inf              0
#> 16:              collate_fn ParamUty    NA    NA     Inf         [NULL]
#> 17:              pin_memory ParamLgl    NA    NA       2          FALSE
#> 18:               drop_last ParamLgl    NA    NA       2          FALSE
#> 19:                 timeout ParamDbl  -Inf   Inf     Inf             -1
#> 20:          worker_init_fn ParamUty    NA    NA     Inf <NoDefault[0]>
#> 21:          worker_globals ParamUty    NA    NA     Inf <NoDefault[0]>
#> 22:         worker_packages ParamUty    NA    NA     Inf <NoDefault[0]>
#> 23:          tensor_dataset ParamFct    NA    NA       1 <NoDefault[0]>
#> 24:               jit_trace ParamLgl    NA    NA       2 <NoDefault[0]>
#> 25:                 neurons ParamUty    NA    NA     Inf <NoDefault[0]>
#> 26:                       p ParamDbl 0e+00 1e+00     Inf <NoDefault[0]>
#> 27:                n_layers ParamInt 1e+00   Inf     Inf <NoDefault[0]>
#> 28:              activation ParamUty    NA    NA     Inf <NoDefault[0]>
#> 29:         activation_args ParamUty    NA    NA     Inf <NoDefault[0]>
#> 30:                   shape ParamUty    NA    NA     Inf <NoDefault[0]>
#> 31:                  opt.lr ParamDbl 0e+00   Inf     Inf          0.001
#> 32:               opt.betas ParamUty    NA    NA     Inf    0.900,0.999
#> 33:                 opt.eps ParamDbl 1e-16 1e-04     Inf          1e-08
#> 34:        opt.weight_decay ParamDbl 0e+00 1e+00     Inf              0
#> 35:             opt.amsgrad ParamLgl    NA    NA       2          FALSE
#> 36:        opt.param_groups ParamUty    NA    NA     Inf <NoDefault[0]>
#> 37:          loss.reduction ParamFct    NA    NA       2           mean
#> 38:      cb.checkpoint.path ParamUty    NA    NA     Inf <NoDefault[0]>
#> 39:      cb.checkpoint.freq ParamInt 1e+00   Inf     Inf <NoDefault[0]>
#> 40: cb.checkpoint.freq_type ParamFct    NA    NA       2          epoch
#>                          id    class lower upper nlevels        default
#>            value
#>           <list>
#>  1:       [NULL]
#>  2:         auto
#>  3:            1
#>  4:            1
#>  5:       random
#>  6:            1
#>  7:    <list[0]>
#>  8:    <list[0]>
#>  9:            0
#> 10:            0
#> 11:       [NULL]
#> 12:         TRUE
#> 13:       [NULL]
#> 14:       [NULL]
#> 15:       [NULL]
#> 16:       [NULL]
#> 17:       [NULL]
#> 18:       [NULL]
#> 19:       [NULL]
#> 20:       [NULL]
#> 21:       [NULL]
#> 22:       [NULL]
#> 23:        FALSE
#> 24:        FALSE
#> 25:             
#> 26:          0.5
#> 27:       [NULL]
#> 28: <nn_relu[1]>
#> 29:    <list[0]>
#> 30:       [NULL]
#> 31:       [NULL]
#> 32:       [NULL]
#> 33:       [NULL]
#> 34:       [NULL]
#> 35:       [NULL]
#> 36:       [NULL]
#> 37:       [NULL]
#> 38:       [NULL]
#> 39:       [NULL]
#> 40:       [NULL]
#>            value
```
