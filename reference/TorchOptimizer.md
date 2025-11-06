# Torch Optimizer

This wraps a `torch::torch_optimizer_generator` and annotates it with
metadata, most importantly a
[`ParamSet`](https://paradox.mlr-org.com/reference/ParamSet.html). The
optimizer is created for the given parameter values by calling the
`$generate()` method.

This class is usually used to configure the optimizer of a torch
learner, e.g. when constructing a learner or in a
[`ModelDescriptor`](https://mlr3torch.mlr-org.com/reference/ModelDescriptor.md).

For a list of available optimizers, see
[`mlr3torch_optimizers`](https://mlr3torch.mlr-org.com/reference/mlr3torch_optimizers.md).
Items from this dictionary can be retrieved using
[`t_opt()`](https://mlr3torch.mlr-org.com/reference/t_opt.md).

## Parameters

Defined by the constructor argument `param_set`. If no parameter set is
provided during construction, the parameter set is constructed by
creating a parameter for each argument of the wrapped loss function,
where the parameters are then of type
[`ParamUty`](https://paradox.mlr-org.com/reference/Domain.html).

## See also

Other Torch Descriptor:
[`TorchCallback`](https://mlr3torch.mlr-org.com/reference/TorchCallback.md),
[`TorchDescriptor`](https://mlr3torch.mlr-org.com/reference/TorchDescriptor.md),
[`TorchLoss`](https://mlr3torch.mlr-org.com/reference/TorchLoss.md),
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
-\> `TorchOptimizer`

## Methods

### Public methods

- [`TorchOptimizer$new()`](#method-TorchOptimizer-new)

- [`TorchOptimizer$generate()`](#method-TorchOptimizer-generate)

- [`TorchOptimizer$clone()`](#method-TorchOptimizer-clone)

Inherited methods

- [`mlr3torch::TorchDescriptor$help()`](https://mlr3torch.mlr-org.com/reference/TorchDescriptor.html#method-help)
- [`mlr3torch::TorchDescriptor$print()`](https://mlr3torch.mlr-org.com/reference/TorchDescriptor.html#method-print)

------------------------------------------------------------------------

### Method `new()`

Creates a new instance of this
[R6](https://r6.r-lib.org/reference/R6Class.html) class.

#### Usage

    TorchOptimizer$new(
      torch_optimizer,
      param_set = NULL,
      id = NULL,
      label = NULL,
      packages = NULL,
      man = NULL
    )

#### Arguments

- `torch_optimizer`:

  (`torch_optimizer_generator`)  
  The torch optimizer.

- `param_set`:

  (`ParamSet` or `NULL`)  
  The parameter set. If `NULL` (default) it is inferred from
  `torch_optimizer`.

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

### Method `generate()`

Instantiates the optimizer.

#### Usage

    TorchOptimizer$generate(params)

#### Arguments

- `params`:

  (named [`list()`](https://rdrr.io/r/base/list.html) of
  [`torch_tensor`](https://torch.mlverse.org/docs/reference/torch_tensor.html)s)  
  The parameters of the network.

#### Returns

`torch_optimizer`

------------------------------------------------------------------------

### Method `clone()`

The objects of this class are cloneable with this method.

#### Usage

    TorchOptimizer$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.

## Examples

``` r
# Create a new torch optimizer
torch_opt = TorchOptimizer$new(optim_ignite_adam, label = "adam")
torch_opt
#> <TorchOptimizer:optim_ignite_adam> adam
#> * Generator: optim_ignite_adam
#> * Parameters: list()
#> * Packages: torch,mlr3torch
# If the param set is not specified, parameters are inferred but are of class ParamUty
torch_opt$param_set
#> <ParamSet(6)>
#>              id    class lower upper nlevels        default  value
#>          <char>   <char> <num> <num>   <num>         <list> <list>
#> 1:           lr ParamUty    NA    NA     Inf <NoDefault[0]> [NULL]
#> 2:        betas ParamUty    NA    NA     Inf <NoDefault[0]> [NULL]
#> 3:          eps ParamUty    NA    NA     Inf <NoDefault[0]> [NULL]
#> 4: weight_decay ParamUty    NA    NA     Inf <NoDefault[0]> [NULL]
#> 5:      amsgrad ParamUty    NA    NA     Inf <NoDefault[0]> [NULL]
#> 6: param_groups ParamUty    NA    NA     Inf <NoDefault[0]> [NULL]

# open the help page of the wrapped optimizer
# torch_opt$help()

# Retrieve an optimizer from the dictionary
torch_opt = t_opt("sgd", lr = 0.1)
torch_opt
#> <TorchOptimizer:sgd> Stochastic Gradient Descent
#> * Generator: optim_ignite_sgd
#> * Parameters: lr=0.1
#> * Packages: torch,mlr3torch
torch_opt$param_set
#> <ParamSet(6)>
#>              id    class lower upper nlevels        default  value
#>          <char>   <char> <num> <num>   <num>         <list> <list>
#> 1:           lr ParamDbl     0   Inf     Inf <NoDefault[0]>    0.1
#> 2:     momentum ParamDbl     0     1     Inf              0 [NULL]
#> 3:    dampening ParamDbl     0     1     Inf              0 [NULL]
#> 4: weight_decay ParamDbl     0     1     Inf              0 [NULL]
#> 5:     nesterov ParamLgl    NA    NA       2          FALSE [NULL]
#> 6: param_groups ParamUty    NA    NA     Inf <NoDefault[0]> [NULL]
torch_opt$label
#> [1] "Stochastic Gradient Descent"
torch_opt$id
#> [1] "sgd"

# Create the optimizer for a network
net = nn_linear(10, 1)
opt = torch_opt$generate(net$parameters)

# is the same as
optim_sgd(net$parameters, lr = 0.1)
#> <optim_sgd>
#>   Inherits from: <torch_optimizer>
#>   Public:
#>     add_param_group: function (param_group) 
#>     clone: function (deep = FALSE) 
#>     defaults: list
#>     initialize: function (params, lr = optim_required(), momentum = 0, dampening = 0, 
#>     load_state_dict: function (state_dict, ..., .refer_to_state_dict = FALSE) 
#>     param_groups: list
#>     state: State, R6
#>     state_dict: function () 
#>     step: function (closure = NULL) 
#>     zero_grad: function (set_to_none = FALSE) 
#>   Private:
#>     deep_clone: function (name, value) 
#>     step_helper: function (closure, loop_fun) 

# Use in a learner
learner = lrn("regr.mlp", optimizer = t_opt("sgd"))
# The parameters of the optimizer are added to the learner's parameter set
learner$param_set
#> <ParamSetCollection(37)>
#>                      id    class lower upper nlevels        default
#>                  <char>   <char> <num> <num>   <num>         <list>
#>  1:              epochs ParamInt     0   Inf     Inf <NoDefault[0]>
#>  2:              device ParamFct    NA    NA      12 <NoDefault[0]>
#>  3:         num_threads ParamInt     1   Inf     Inf <NoDefault[0]>
#>  4: num_interop_threads ParamInt     1   Inf     Inf <NoDefault[0]>
#>  5:                seed ParamInt  -Inf   Inf     Inf <NoDefault[0]>
#>  6:           eval_freq ParamInt     1   Inf     Inf <NoDefault[0]>
#>  7:      measures_train ParamUty    NA    NA     Inf <NoDefault[0]>
#>  8:      measures_valid ParamUty    NA    NA     Inf <NoDefault[0]>
#>  9:            patience ParamInt     0   Inf     Inf <NoDefault[0]>
#> 10:           min_delta ParamDbl     0   Inf     Inf <NoDefault[0]>
#> 11:          batch_size ParamInt     1   Inf     Inf <NoDefault[0]>
#> 12:             shuffle ParamLgl    NA    NA       2          FALSE
#> 13:             sampler ParamUty    NA    NA     Inf <NoDefault[0]>
#> 14:       batch_sampler ParamUty    NA    NA     Inf <NoDefault[0]>
#> 15:         num_workers ParamInt     0   Inf     Inf              0
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
#> 26:                   p ParamDbl     0     1     Inf <NoDefault[0]>
#> 27:            n_layers ParamInt     1   Inf     Inf <NoDefault[0]>
#> 28:          activation ParamUty    NA    NA     Inf <NoDefault[0]>
#> 29:     activation_args ParamUty    NA    NA     Inf <NoDefault[0]>
#> 30:               shape ParamUty    NA    NA     Inf <NoDefault[0]>
#> 31:              opt.lr ParamDbl     0   Inf     Inf <NoDefault[0]>
#> 32:        opt.momentum ParamDbl     0     1     Inf              0
#> 33:       opt.dampening ParamDbl     0     1     Inf              0
#> 34:    opt.weight_decay ParamDbl     0     1     Inf              0
#> 35:        opt.nesterov ParamLgl    NA    NA       2          FALSE
#> 36:    opt.param_groups ParamUty    NA    NA     Inf <NoDefault[0]>
#> 37:      loss.reduction ParamFct    NA    NA       2           mean
#>                      id    class lower upper nlevels        default
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
```
