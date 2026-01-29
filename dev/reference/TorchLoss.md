# Torch Loss

This wraps a `torch::nn_loss` and annotates it with metadata, most
importantly a
[`ParamSet`](https://paradox.mlr-org.com/reference/ParamSet.html). The
loss function is created for the given parameter values by calling the
`$generate()` method.

This class is usually used to configure the loss function of a torch
learner, e.g. when construcing a learner or in a
[`ModelDescriptor`](https://mlr3torch.mlr-org.com/dev/reference/ModelDescriptor.md).

For a list of available losses, see
[`mlr3torch_losses`](https://mlr3torch.mlr-org.com/dev/reference/mlr3torch_losses.md).
Items from this dictionary can be retrieved using
[`t_loss()`](https://mlr3torch.mlr-org.com/dev/reference/t_loss.md).

## Parameters

Defined by the constructor argument `param_set`. If no parameter set is
provided during construction, the parameter set is constructed by
creating a parameter for each argument of the wrapped loss function,
where the parametes are then of type `ParamUty`.

## See also

Other Torch Descriptor:
[`TorchCallback`](https://mlr3torch.mlr-org.com/dev/reference/TorchCallback.md),
[`TorchDescriptor`](https://mlr3torch.mlr-org.com/dev/reference/TorchDescriptor.md),
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

[`mlr3torch::TorchDescriptor`](https://mlr3torch.mlr-org.com/dev/reference/TorchDescriptor.md)
-\> `TorchLoss`

## Public fields

- `task_types`:

  ([`character()`](https://rdrr.io/r/base/character.html))  
  The task types this loss supports.

## Methods

### Public methods

- [`TorchLoss$new()`](#method-TorchLoss-new)

- [`TorchLoss$print()`](#method-TorchLoss-print)

- [`TorchLoss$generate()`](#method-TorchLoss-generate)

- [`TorchLoss$clone()`](#method-TorchLoss-clone)

Inherited methods

- [`mlr3torch::TorchDescriptor$help()`](https://mlr3torch.mlr-org.com/dev/reference/TorchDescriptor.html#method-help)

------------------------------------------------------------------------

### Method `new()`

Creates a new instance of this
[R6](https://r6.r-lib.org/reference/R6Class.html) class.

#### Usage

    TorchLoss$new(
      torch_loss,
      task_types = NULL,
      param_set = NULL,
      id = NULL,
      label = NULL,
      packages = NULL,
      man = NULL
    )

#### Arguments

- `torch_loss`:

  (`nn_loss` or `function`)  
  The loss module or function that generates the loss module. Can have
  arguments `task` that will be provided when the loss is instantiated.

- `task_types`:

  ([`character()`](https://rdrr.io/r/base/character.html))  
  The task types supported by this loss.

- `param_set`:

  ([`ParamSet`](https://paradox.mlr-org.com/reference/ParamSet.html) or
  `NULL`)  
  The parameter set. If `NULL` (default) it is inferred from
  `torch_loss`.

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

------------------------------------------------------------------------

### Method [`print()`](https://rdrr.io/r/base/print.html)

Prints the object

#### Usage

    TorchLoss$print(...)

#### Arguments

- `...`:

  any

------------------------------------------------------------------------

### Method `generate()`

Instantiates the loss function.

#### Usage

    TorchLoss$generate(task = NULL)

#### Arguments

- `task`:

  (`Task`)  
  The task. Must be provided if the loss function requires a task.

#### Returns

`torch_loss`

------------------------------------------------------------------------

### Method `clone()`

The objects of this class are cloneable with this method.

#### Usage

    TorchLoss$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.

## Examples

``` r
# Create a new torch loss
torch_loss = TorchLoss$new(torch_loss = nn_mse_loss, task_types = "regr")
torch_loss
#> <TorchLoss:nn_mse_loss> nn_mse_loss
#> * Generator: nn_mse_loss
#> * Parameters: list()
#> * Packages: torch,mlr3torch
#> * Task Types: regr
# the parameters are inferred
torch_loss$param_set
#> <ParamSet(1)>
#>           id    class lower upper nlevels        default  value
#>       <char>   <char> <num> <num>   <num>         <list> <list>
#> 1: reduction ParamUty    NA    NA     Inf <NoDefault[0]> [NULL]

# Retrieve a loss from the dictionary:
torch_loss = t_loss("mse", reduction = "mean")
# is the same as
torch_loss
#> <TorchLoss:mse> Mean Squared Error
#> * Generator: nn_mse_loss
#> * Parameters: reduction=mean
#> * Packages: torch,mlr3torch
#> * Task Types: regr
torch_loss$param_set
#> <ParamSet(1)>
#>           id    class lower upper nlevels default  value
#>       <char>   <char> <num> <num>   <num>  <list> <list>
#> 1: reduction ParamFct    NA    NA       2    mean   mean
torch_loss$label
#> [1] "Mean Squared Error"
torch_loss$task_types
#> [1] "regr"
torch_loss$id
#> [1] "mse"

# Create the loss function
loss_fn = torch_loss$generate()
loss_fn
#> An `nn_module` containing 0 parameters.
# Is the same as
nn_mse_loss(reduction = "mean")
#> An `nn_module` containing 0 parameters.

# open the help page of the wrapped loss function
# torch_loss$help()

# Use in a learner
learner = lrn("regr.mlp", loss = t_loss("mse"))
# The parameters of the loss are added to the learner's parameter set
learner$param_set
#> <ParamSetCollection(37)>
#>                      id    class lower upper nlevels        default
#>                  <char>   <char> <num> <num>   <num>         <list>
#>  1:              epochs ParamInt 0e+00   Inf     Inf <NoDefault[0]>
#>  2:              device ParamFct    NA    NA      12 <NoDefault[0]>
#>  3:         num_threads ParamInt 1e+00   Inf     Inf <NoDefault[0]>
#>  4: num_interop_threads ParamInt 1e+00   Inf     Inf <NoDefault[0]>
#>  5:                seed ParamInt  -Inf   Inf     Inf <NoDefault[0]>
#>  6:           eval_freq ParamInt 1e+00   Inf     Inf <NoDefault[0]>
#>  7:      measures_train ParamUty    NA    NA     Inf <NoDefault[0]>
#>  8:      measures_valid ParamUty    NA    NA     Inf <NoDefault[0]>
#>  9:            patience ParamInt 0e+00   Inf     Inf <NoDefault[0]>
#> 10:           min_delta ParamDbl 0e+00   Inf     Inf <NoDefault[0]>
#> 11:          batch_size ParamInt 1e+00   Inf     Inf <NoDefault[0]>
#> 12:             shuffle ParamLgl    NA    NA       2          FALSE
#> 13:             sampler ParamUty    NA    NA     Inf <NoDefault[0]>
#> 14:       batch_sampler ParamUty    NA    NA     Inf <NoDefault[0]>
#> 15:         num_workers ParamInt 0e+00   Inf     Inf              0
#> 16:          collate_fn ParamUty    NA    NA     Inf         [NULL]
#> 17:          pin_memory ParamLgl    NA    NA       2          FALSE
#> 18:           drop_last ParamLgl    NA    NA       2          FALSE
#> 19:             timeout ParamDbl  -Inf   Inf     Inf             -1
#> 20:      worker_init_fn ParamUty    NA    NA     Inf <NoDefault[0]>
#> 21:      worker_globals ParamUty    NA    NA     Inf <NoDefault[0]>
#> 22:     worker_packages ParamUty    NA    NA     Inf <NoDefault[0]>
#> 23:      tensor_dataset ParamFct    NA    NA       1 <NoDefault[0]>
#> 24:           jit_trace ParamLgl    NA    NA       2 <NoDefault[0]>
#> 25:             neurons ParamUty    NA    NA     Inf <NoDefault[0]>
#> 26:                   p ParamDbl 0e+00 1e+00     Inf <NoDefault[0]>
#> 27:            n_layers ParamInt 1e+00   Inf     Inf <NoDefault[0]>
#> 28:          activation ParamUty    NA    NA     Inf <NoDefault[0]>
#> 29:     activation_args ParamUty    NA    NA     Inf <NoDefault[0]>
#> 30:               shape ParamUty    NA    NA     Inf <NoDefault[0]>
#> 31:              opt.lr ParamDbl 0e+00   Inf     Inf          0.001
#> 32:           opt.betas ParamUty    NA    NA     Inf    0.900,0.999
#> 33:             opt.eps ParamDbl 1e-16 1e-04     Inf          1e-08
#> 34:    opt.weight_decay ParamDbl 0e+00 1e+00     Inf              0
#> 35:         opt.amsgrad ParamLgl    NA    NA       2          FALSE
#> 36:    opt.param_groups ParamUty    NA    NA     Inf <NoDefault[0]>
#> 37:      loss.reduction ParamFct    NA    NA       2           mean
#>                      id    class lower upper nlevels        default
#>                  <char>   <char> <num> <num>   <num>         <list>
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
#>            value
#>           <list>
```
